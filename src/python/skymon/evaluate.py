"""
Compare stored candidates with the simulator's truth file.

A simulated FRB is recovered by a candidate within match_radius_deg on the
sky whose DM is within 10% (or three DM steps) and whose bottom-edge arrival
time agrees within three tier samples plus twice the boxcar width plus 2 s.
Candidates matching no FRB and not marked as RFI or as a live injection are
false candidates.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from .candidates import separation_deg


def load_candidates(out_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(out_dir, "candidates", "*", "*.json"))):
        with open(p) as f:
            c = json.load(f)
        if c.get("files", {}).get("superseded_by"):
            continue
        out.append(c)
    return out


def load_truth(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _matches(t, c, radius_deg):
    from .candidates import elevation_radius
    r = float(elevation_radius(radius_deg, c.get("el", 90.0)))
    return (separation_deg(t["ra"], t["dec"], c["ra"], c["dec"]) <= r
            and abs(c["dm"] - t["dm"]) <= max(0.1 * t["dm"], 3 * c["dm_step"])
            and abs(c["t_bottom"] - t["t_bottom"]) <= 3 * c["tier_dt"] + 2 * c["width_s"] + 2.0)


def match(truth, cands, radius_deg):
    """
    rows: (burst, its brightest matching candidate or None) for each burst.
    false: candidates (not RFI, not injections) that match no burst. A second
    candidate of one burst (another part of its cluster) is not false.
    """
    rows = []
    hit = set()
    for t in truth:
        best = None
        for i, c in enumerate(cands):
            if c["cls"] == "rfi" or not _matches(t, c, radius_deg):
                continue
            hit.add(i)
            if best is None or c["snr"] > cands[best]["snr"]:
                best = i
        rows.append((t, cands[best] if best is not None else None))
    false = [c for i, c in enumerate(cands)
             if i not in hit and c["cls"] not in ("rfi", "injection")]
    return rows, false


def ready_after(tiers, start):
    """
    dm -> unix time from which the tier covering dm reports: its window must
    have filled once for the noise estimate and once more before hits count
    (search.Searcher._search). A monitor that runs continuously passes this
    once, at start-up. tiers: Searcher.describe() of the run.
    """
    def f(dm):
        for t in tiers:
            if dm <= t["dm_max"]:
                return start + (t["window"] + 2 * t["block"]) * t["dt_s"]
        t = tiers[-1]
        return start + (t["window"] + 2 * t["block"]) * t["dt_s"]
    return f


def report(truth_path, out_dir, radius_deg=None, t_end=None, latency_s=0.0, ready=None):
    """
    ready: optional dm -> unix time from which a burst can be found (see
    ready_after); bursts arriving at the bottom of the band earlier are left
    out of the count, as are bursts not searched before t_end.
    """
    truth = load_truth(truth_path)
    all_truth = list(truth)
    cands = load_candidates(out_dir)
    if radius_deg is None:
        f = np.mean([c["f_lo_mhz"] + c["f_hi_mhz"] for c in cands]) / 2 if cands else 57.0
        radius_deg = max(2.0 * np.degrees(299.792458 / f / 100.0), 3.0)
    n_all = len(truth)
    if t_end is not None:  # only bursts whose sweep ended early enough to have been searched
        truth = [t for t in truth if t["t_bottom"] + t["width_s"] + latency_s < t_end]
    n_late = n_all - len(truth)
    if ready is not None:  # and not while the tier for their DM was still starting
        truth = [t for t in truth if t["t_bottom"] >= ready(t["dm"])]
    n_warm = n_all - n_late - len(truth)
    rows, _ = match(truth, cands, radius_deg)
    # a candidate is false only if it matches no simulated burst at all,
    # counted or not
    _, false = match(all_truth, cands, radius_deg)
    found = [(t, c) for t, c in rows if c is not None]
    lines = [f"simulated FRBs: {n_all}, of which {n_late} arrived too late to be searched and "
             f"{n_warm} while the tier for their DM was starting"] if (n_late or n_warm) else []
    lines += [f"simulated FRBs counted: {len(truth)}, recovered: {len(found)} "
             f"({100 * len(found) / max(1, len(truth)):.0f}%), false candidates: {len(false)}, "
             f"RFI-classified: {sum(c['cls'] == 'rfi' for c in cands)}"]
    for lo, hi in [(0, 10), (10, 15), (15, 20), (20, 1e9)]:
        sel = [r for r in rows if lo <= r[0]["snr"] < hi]
        if sel:
            k = sum(c is not None for _, c in sel)
            lines.append(f"  target S/N {lo:>2}-{hi if hi < 1e9 else 'inf'}: {k}/{len(sel)}")
    for lo, hi in [(0, 100), (100, 300), (300, 1000), (1000, 1e9)]:
        sel = [r for r in rows if lo <= r[0]["dm"] < hi]
        if sel:
            k = sum(c is not None for _, c in sel)
            lines.append(f"  DM {lo:>4}-{hi if hi < 1e9 else 'inf'}: {k}/{len(sel)}")
    if found:
        ratio = np.array([c["snr"] / t["snr"] for t, c in found])
        ddm = np.array([(c["dm"] - t["dm"]) / t["dm"] for t, c in found])
        lines.append(f"  measured/target S/N: median {np.median(ratio):.2f} "
                     f"(16-84%: {np.percentile(ratio, 16):.2f}-{np.percentile(ratio, 84):.2f}); "
                     f"DM error median {100 * np.median(np.abs(ddm)):.1f}%")
    return dict(text="\n".join(lines), rows=rows, false=false, truth=truth, cands=cands)
