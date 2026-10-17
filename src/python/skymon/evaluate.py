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


def load_events(combiner_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(combiner_dir, "events", "*", "*.json"))):
        with open(p) as f:
            out.append(json.load(f))
    return out


def report_combiner(frb_file, combiner_dir, radius_deg=6.0):
    """
    Compare the combiner's events with the FRBs given to the simulators of all
    bands (simulate.make_frb_list). An FRB is found when an event matches its
    position, DM (10%) and t_inf; the event's result says whether the bands
    agreed. Events matching no FRB and no coordinated injection are listed as
    unexplained.
    """
    from .dedisp import K_DM

    with open(frb_file) as f:
        frbs = json.load(f)
    events = load_events(combiner_dir)
    rows, used = [], set()
    for fr in frbs:
        best = None
        for i, e in enumerate(events):
            if e["kind"] != "sky":
                continue
            if separation_deg(fr["ra"], fr["dec"], e["ra"], e["dec"]) > radius_deg * 2:
                continue
            if abs(e["dm"] - fr["dm"]) > max(0.1 * fr["dm"], 3 * e["dm_err"]):
                continue
            # t_inf is uncertain by the DM error times the delay at the reference band
            tol = 30.0 + K_DM * 3 * e["dm_err"] / e["ref_f_lo"] ** 2
            if abs(e["t_inf"] - fr["t_inf"]) > tol:
                continue
            best = i
            break
        if best is not None:
            used.add(best)
        rows.append((fr, events[best] if best is not None else None))
    unexplained = [e for i, e in enumerate(events) if i not in used and e["kind"] == "sky"]
    inj = [e for e in events if e["kind"] == "injection"]
    lines = [f"FRBs given to the simulators: {len(frbs)}; with an event: "
             f"{sum(e is not None for _, e in rows)}"]
    for st in ("confirmed", "band_limited", "implausible", "unconfirmable", "pending"):
        n = sum(1 for _, e in rows if e is not None and e["status"] == st)
        if n:
            lines.append(f"  {st}: {n}")
    errs = [abs(e["dm"] - fr["dm"]) / fr["dm"] for fr, e in rows
            if e is not None and e["status"] == "confirmed"]
    if errs:
        lines.append(f"  DM error of confirmed events: median {100 * np.median(errs):.2f}%, "
                     f"max {100 * np.max(errs):.2f}%")
    lines.append(f"events matching no FRB: {len(unexplained)} ("
                 + ", ".join(f"{st} {sum(e['status'] == st for e in unexplained)}"
                             for st in ("confirmed", "band_limited", "implausible", "unconfirmable",
                                        "pending"))
                 + ")")
    lines.append("coordinated injections: " + (", ".join(
        f"{st} {sum(e['status'] == st for e in inj)}"
        for st in ("confirmed", "band_limited", "implausible", "unconfirmable", "pending"))
        if inj else "none"))
    return dict(text="\n".join(lines), rows=rows, unexplained=unexplained, injections=inj,
                events=events)


def report_no_loss(frb_file, combiner_dir, monitor_dirs, detectable_snr=10.0):
    """
    Does any burst that a band could detect go unrecorded? For each simulated
    FRB, the expected S/N in each band (after its spectrum and scattering) is
    read from that band's truth file. An FRB is detectable if some band
    expects S/N >= detectable_snr. It is kept if the combiner has an event for
    it, whatever the event's result, since no result discards data. For each
    kept FRB, the bands with data on disk (detection, followup, or context)
    are counted.
    """
    base = report_combiner(frb_file, combiner_dir)
    expected = {}
    for name, d in monitor_dirs.items():
        p = os.path.join(d, "truth.jsonl")
        if not os.path.exists(p):
            continue
        for t in load_truth(p):
            expected.setdefault(t["id"], {})[name] = t.get("meta", {}).get("snr_expected",
                                                                           t["snr"])
    lines, lost, kept = [], [], []
    by_status = {}
    for fr, ev in base["rows"]:
        exp = expected.get(fr["id"], {})
        if not exp or max(exp.values()) < detectable_snr:
            continue
        if ev is None:
            lost.append((fr, exp))
            continue
        kept.append((fr, ev, exp))
        by_status[ev["status"]] = by_status.get(ev["status"], 0) + 1
    n_det = len(lost) + len(kept)
    lines.append(f"FRBs detectable in at least one band (expected S/N >= {detectable_snr:g}): "
                 f"{n_det}; kept as events: {len(kept)}; lost: {len(lost)}")
    lines.append("  results of the kept ones: " + ", ".join(f"{k} {v}" for k, v in
                                                          sorted(by_status.items())))
    with_data = [sum(1 for b in ev["bands"].values()
                     if b.get("state") == "detected" or b.get("data_id")) for _, ev, _ in kept]
    if with_data:
        lines.append(f"  bands with data on disk per kept event: median {np.median(with_data):.0f}, "
                     f"min {min(with_data)}")
    for fr, exp in lost:
        lines.append(f"  lost: {fr['id']} DM {fr['dm']:.0f} expected S/N "
                     + ", ".join(f"{k} {v:.1f}" for k, v in exp.items()))
    return dict(text="\n".join(lines + [base["text"]]), lost=lost, kept=kept, base=base)
