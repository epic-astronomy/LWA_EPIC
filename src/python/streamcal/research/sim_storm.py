"""Closed-loop calibration through an ionospheric storm (voltage-tap route only).

Truth sky per 10 s interval:
  diffuse LFSM (nside 128, 10% small-scale structure) and 25 faint sources, fixed per minute
  bright sources (A-team, Sun) at the model positions shifted by refraction
      d(t, f) = A (74 MHz / f)^2 [cos(2 pi t / P + p_k), sin(2 pi t / P + p_k)]   (a TID)
  with fluxes multiplied by independent log-normal scintillation factors per interval,
  a solar burst (Sun x burst_factor) and whole-sky absorption over set windows.
True gains: amplitude scatter, phase offsets, delays, slow random walks.

Solver variants, each feeding its own Kalman tracker and applied gains:
  A  model fixed except the Sun flux (the daemon as first pushed)
  B  all bright-source fluxes free
  C  fluxes and per-channel tangent-plane offsets free
  D  C, then offsets fitted to d = d0 (f0/f)^2 across the band and the band re-solved
     with the positions fixed to the fit (fluxes free)
Errors are reported with the image shift (linear phase gradient) removed, and the
shift separately.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from astropy.time import Time, TimeDelta  # noqa: E402

from streamcal import epicop as E, forward as F, solvers as C  # noqa: E402
import simutil as U  # noqa: E402

G = {}
VARIANTS = ["A", "B", "C", "D"]


def worker(args):
    ci, k_snap, eps, truth_pts, applied, fixed_pos, seed, phase, absorb = args
    rng = np.random.default_rng(seed)
    f = G["freq"][ci]
    r = G["r"]
    n = len(r)
    Mt = G["Mt_fixed"][k_snap][ci].copy()
    for enu, app in truth_pts[ci]:
        Mt += C.point_matrix(r, f, enu, app)
    Mt *= absorb
    Mt += G["noise"][ci] * np.eye(n)
    Rt = eps[:, None] * Mt * eps.conj()[None, :]
    T = max(int(G["duty"] * G["dt"] * E.CHAN_BW), n + 1)
    R = U.wishart(Rt, T, rng)
    W = G["W"][ci]
    Md = G["Mm_diff"][k_snap][ci]
    srcs = G["srcs"][k_snap][ci]
    out = {}
    for v in (VARIANTS if phase == 1 else ["D"]):
        g0 = applied[v].copy()
        g0[~np.isfinite(g0) | (np.abs(g0) < 1e-3)] = 1.0
        if v == "A":
            fixed = Md + sum(C.point_matrix(r, f, s[0], s[1]) for s in srcs if s[2] != "Sun")
            sun = [C.point_matrix(r, f, s[0], s[1]) for s in srcs if s[2] == "Sun"]
            g, _, _ = C.stefcal(R, fixed, W, g0=g0, niter=40, free=sun or None)
            par = None
        elif v == "B":
            g, sc, off, _ = C.stefcal_sky(R, Md, [(s[0], s[1]) for s in srcs], W, r, f, g0=g0, solve_pos=False)
            par = (sc, off)
        elif v == "C" or (v == "D" and phase == 1):
            g, sc, off, _ = C.stefcal_sky(R, Md, [(s[0], s[1]) for s in srcs], W, r, f, g0=g0, solve_pos=True)
            par = (sc, off)
        else:  # D, second pass with positions from the band fit
            g, sc, off, _ = C.stefcal_sky(R, Md, [(s[0], s[1]) for s in srcs], W, r, f, g0=g0, solve_pos=False,
                                          fixed_pos=fixed_pos)
            par = (sc, off)
        # Kalman measurement variance: residual about the diffuse part of the model
        # (the same rough noise level for every variant), over each stand's sensitivity
        den = np.sum(np.abs(g[None, :]) ** 2 * np.abs(Md) ** 2 * W, axis=1) + 1e-30
        sig2 = np.sum(np.abs((R - np.outer(g, g.conj()) * Md) * W) ** 2) / W.sum()
        out[v] = (g, sig2 / den, par)
    return ci, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chan0", type=int, default=3104)
    ap.add_argument("--nchan", type=int, default=16)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--duration", type=float, default=600)
    ap.add_argument("--dt", type=float, default=10.0)
    ap.add_argument("--duty", type=float, default=0.05)
    ap.add_argument("--start", default="2026-10-16T06:00:00")
    ap.add_argument("--refr_deg", type=float, default=0.3, help="refraction amplitude at 74 MHz")
    ap.add_argument("--tid_period", type=float, default=600.0)
    ap.add_argument("--scint", type=float, default=0.3, help="log-normal sigma of flux per interval")
    ap.add_argument("--burst", default="200:260:30", help="Sun burst start:end:factor (s)")
    ap.add_argument("--absorb", default="400:500:0.7", help="whole-sky absorption start:end:factor (s)")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--out", default="sim_storm.npz")
    a = ap.parse_args()
    cfg = E.EpicConfig()
    chans = a.chan0 + a.step * np.arange(a.nchan)
    freq = chans * E.CHAN_BW
    t0 = Time(a.start)
    ant = np.flatnonzero(E.eg.used_stands())
    r = F.stand_enu()[ant]
    n = len(ant)
    bl = np.linalg.norm(r[:, None, :2] - r[None, :, :2], axis=-1)
    G.update(freq=freq, r=r, dt=a.dt, duty=a.duty)
    G["W"] = []
    for f in freq:
        W = (bl >= 299792458.0 / f).astype(float)
        np.fill_diagonal(W, 0)
        G["W"].append(W)
    ex = F.random_extra_sources()
    snap_t = np.arange(0, a.duration + 60, 60.0)
    G["Mt_fixed"], G["Mm_diff"], G["srcs"], G["noise"] = [], [], [], None
    T0 = time.time()
    for ts in snap_t:
        t = t0 + TimeDelta(ts, format="sec")
        tr = F.build_snapshot(t, chans, nside=128, extra_sources=ex, fluct_rms=0.1)
        mo = F.build_snapshot(t, chans, nside=64)
        G["Mt_fixed"].append([tr.matrix(ant, i, names=["diffuse", "extra"]) for i in range(len(chans))])
        G["Mm_diff"].append([mo.matrix(ant, i, names=["diffuse"]) for i in range(len(chans))])
        G["srcs"].append([[(c.enu[0], c.flux[0, i], c.name) for c in mo.comps if c.name not in ("diffuse", "extra")]
                          for i in range(len(chans))])
        if G["noise"] is None:
            G["noise"] = [0.15 * np.real(np.diag(G["Mt_fixed"][0][i])).mean() for i in range(len(chans))]
        print(f"snapshot t+{ts:.0f}s ({time.time() - T0:.0f}s), bright sources {[s[2] for s in G['srcs'][-1][0]]}", flush=True)

    rng = np.random.default_rng(11)
    t_grid = np.arange(0, a.duration, a.dt)
    amp0 = np.exp(rng.normal(0, 0.2, n))
    ph0 = rng.uniform(-1, 1, n) * np.radians(70)
    tau0 = rng.normal(0, 3.0, n)
    nu_c = freq.mean()
    walk_p = np.cumsum(rng.normal(0, np.radians(1) * np.sqrt(a.dt / 60), (len(t_grid), n)), 0)
    walk_t = np.cumsum(rng.normal(0, 0.02 * np.sqrt(a.dt / 60), (len(t_grid), n)), 0)
    b0, b1, bf = (float(x) for x in a.burst.split(":"))
    s0, s1, sf = (float(x) for x in a.absorb.split(":"))
    names_all = sorted({s[2] for snap in G["srcs"] for s in snap[0]})
    tid_phase = {nm: rng.uniform(0, 2 * np.pi) for nm in names_all}
    applied = {v: np.ones((len(chans), n), complex) for v in VARIANTS}
    trackers = {v: C.Tracker(n, freq, nu_c, q_phase=np.radians(0.5) ** 2, q_delay=0.02 ** 2) for v in VARIANTS}
    rec = {v: [] for v in VARIANTS}
    raw = {v: [] for v in VARIANTS}
    rec_none, offs_rec = [], []
    ctx = mp.get_context("fork")
    with ctx.Pool(a.procs) as pool:
        for k, ts in enumerate(t_grid):
            k_snap = int(ts // 60)
            eps = np.empty((len(chans), n), complex)
            for ci, f in enumerate(freq):
                eps[ci] = amp0 * np.exp(1j * (ph0 + walk_p[k] + 2 * np.pi * (f - nu_c) * 1e-9 * (tau0 + walk_t[k])))
            # truth bright sources this interval
            truth_pts, true_off = [], {}
            scint = {nm: np.exp(rng.normal(0, a.scint)) for nm in names_all}
            for ci, f in enumerate(freq):
                pts = []
                for enu, app, nm in G["srcs"][k_snap][ci]:
                    amp = np.radians(a.refr_deg) * (74e6 / f) ** 2
                    d = amp * np.array([np.cos(2 * np.pi * ts / a.tid_period + tid_phase[nm]),
                                        np.sin(2 * np.pi * ts / a.tid_period + tid_phase[nm])])
                    fac = scint[nm] * (bf if (nm == "Sun" and b0 <= ts < b1) else 1.0)
                    pts.append((C.shift_direction(enu, d), app * fac))
                    if ci == 0:
                        true_off[nm] = d
                truth_pts.append(pts)
            absorb = sf if s0 <= ts < s1 else 1.0
            jobs = [(ci, k_snap, eps[ci], truth_pts, {v: applied[v][ci] for v in VARIANTS}, None, 1000 * k + ci, 1, absorb)
                    for ci in range(len(chans))]
            res = dict(pool.map(worker, jobs))
            # D: fit offsets ~ (f0/f)^2 per source, then re-solve with fixed positions
            K = len(G["srcs"][k_snap][0])
            offC = np.array([res[ci]["D"][2][1] for ci in range(len(chans))])  # [chan, K, 2]
            scC = np.array([res[ci]["D"][2][0] for ci in range(len(chans))])
            x = (freq.mean() / freq) ** 2
            wts = np.clip(scC, 0, None)  # weight by solved flux (sources that vanished carry no position)
            d0 = np.zeros((K, 2))
            for kk in range(K):
                w = wts[:, kk] + 1e-9
                for j in range(2):
                    d0[kk, j] = np.sum(w * x * offC[:, kk, j]) / np.sum(w * x * x)
            jobs2 = [(ci, k_snap, eps[ci], truth_pts, {v: applied[v][ci] for v in VARIANTS},
                      d0 * x[ci], 1000 * k + ci, 2, absorb) for ci in range(len(chans))]
            res2 = dict(pool.map(worker, jobs2))
            for ci in range(len(chans)):
                res[ci]["D"] = res2[ci]["D"]
            offs_rec.append(dict(fit=d0, true={nm: true_off[nm] * (74e6 / freq.mean()) ** 2 / (74e6 / freq[0]) ** 2
                                               for nm in true_off}))
            line = []
            for v in VARIANTS:
                g = np.array([res[ci][v][0] for ci in range(len(chans))]).T
                var = np.array([res[ci][v][1] for ci in range(len(chans))]).T
                raw[v].append(np.mean([U.gain_error_tiptilt(g[:, ci], eps[ci], r, freq[ci]) for ci in range(len(chans))], 0))
                trackers[v].update(g, var, a.dt)
                applied[v] = trackers[v].gains().T
                e = np.mean([U.gain_error_tiptilt(applied[v][ci], eps[ci], r, freq[ci]) for ci in range(len(chans))], 0)
                rec[v].append(e)
                line.append(f"{v} {raw[v][-1][1]:5.2f}/{e[1]:5.2f}deg sh{e[2]:.2f}")
            e0 = np.mean([U.gain_error_tiptilt(np.ones(n), eps[ci], r, freq[ci]) for ci in range(len(chans))], 0)
            rec_none.append(e0)
            ev = ("burst " if b0 <= ts < b1 else "") + ("absorb " if s0 <= ts < s1 else "")
            print(f"t={ts:4.0f}s {ev:13s} " + " | ".join(line) + f"  ({time.time() - T0:.0f}s)", flush=True)
    np.savez(a.out, t=t_grid, freq=freq, none=np.array(rec_none), **{f"err_{v}": np.array(rec[v]) for v in VARIANTS},
             **{f"raw_{v}": np.array(raw[v]) for v in VARIANTS}, args=str(vars(a)))


if __name__ == "__main__":
    main()
