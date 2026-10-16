"""Closed-loop streaming calibration simulation for one EPIC band (X polarisation).

Every interval (dt seconds) the "instrument" produces, per channel, the
full-rate sample covariance (what the GPU images) and a duty-cycled tap
covariance (what a CPU X-engine would see). Each method solves for gains,
feeds a tracker, and the tracker's gain table is what the imager applies in
the next interval. Methods:

  none     cable model only (gains = 1)
  tap      voltage tap -> CPU StEFCal (forward-modelled sky) -> Kalman
  image    EPIC images -> grid-domain solver (forward-modelled sky) -> Kalman
  epical   Beardsley et al. 2017 feedback on the Cyg A pixel, point-source model

Outputs an npz with per-interval gain errors and image fidelity metrics.
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streamcal import sky as skymodel, epicop, forward, solvers as calib  # noqa: E402,F401
sys.modules.update({"skymodel": skymodel, "epicop": epicop, "forward": forward, "calib": calib})


import argparse
import multiprocessing as mp
import time

import numpy as np
from astropy.time import Time, TimeDelta

import forward as F
import epicop as E
import calib as C
import simutil as U

G = {}  # per-process globals (filled before fork)


def true_gains(rng, n, nu, nu_c, t_grid):
    """Time-variable true residual gains eps[t, chan, ant] relative to the cable model."""
    nt, nc = len(t_grid), len(nu)
    amp0 = np.exp(rng.normal(0, 0.2, n))
    ripple_a = 1 + 0.05 * np.sin(2 * np.pi * (nu[:, None] - nu[0]) / 1.1e6 + rng.uniform(0, 2 * np.pi, n))
    ripple_p = np.radians(5) * np.sin(2 * np.pi * (nu[:, None] - nu[0]) / 0.8e6 + rng.uniform(0, 2 * np.pi, n))
    ph0 = rng.uniform(-np.pi, np.pi, n) * 0.4  # up to +-72 deg
    tau0 = rng.normal(0, 3.0, n)  # ns
    dt = np.diff(t_grid, prepend=t_grid[0])
    ph_walk = np.cumsum(rng.normal(0, np.radians(3) / np.sqrt(60), (nt, n)) * np.sqrt(dt)[:, None], 0)
    tau_walk = np.cumsum(rng.normal(0, 0.05 / np.sqrt(60), (nt, n)) * np.sqrt(dt)[:, None], 0)
    amp_drift = 1 + 0.01 * np.outer(t_grid / 600.0, rng.normal(0, 1, n))
    jump = np.zeros((nt, n))
    jumpers = rng.choice(n, 4, replace=False)
    jump[np.ix_(t_grid >= 300, jumpers)] = np.radians(120)
    dead = rng.choice(np.setdiff1d(np.arange(n), jumpers), 1)
    deadf = np.ones((nt, n))
    deadf[np.ix_(t_grid >= 450, dead)] = 0.02
    eps = np.empty((nt, nc, n), complex)
    for k in range(nt):
        ph = ph0 + ph_walk[k] + jump[k] + 2 * np.pi * (nu[:, None] - nu_c) * 1e-9 * (tau0 + tau_walk[k]) + ripple_p
        eps[k] = (amp0 * amp_drift[k] * deadf[k] * ripple_a) * np.exp(1j * ph)
    return eps, dict(jumpers=jumpers, dead=dead)


def pixel_beam(op, ci, enu_src):
    """Holographic E-field beam k_b of each used stand toward a direction (EPICal pixel)."""
    chan = G["chans"][ci]
    kc, xp, yp, pos, used = E.kernels(G["cfg"], chan)
    f = chan * E.CHAN_BW
    p, Fr, _ = E.eg.array_frame()
    s = Fr @ enu_src  # direction in grid frame
    lm = s[:2]
    N, delta = G["cfg"].grid_size, G["cfg"].delta
    k = np.zeros(len(op.ant), complex)
    for i, a in enumerate(op.ant):
        # E-field image pixel = sum over grid cells of kernel * exp(2 pi i (cell * delta) . lm)
        k[i] = np.sum(kc[a] * np.exp(-2j * np.pi * delta * (xp[a] * lm[0] + yp[a] * lm[1])))
    return k


def worker_init():
    pass


def solve_channel(args):
    ci, k_int, k_snap, applied, eps, seed, want_img = args
    rng = np.random.default_rng(seed)
    op = G["ops"][ci]
    n = len(op.ant)
    phi = G["phi"][ci]
    Mt = G["Mt"][k_snap][ci]
    Mm = G["Mm"][k_snap][ci]
    Mpt = G["Mpt"][k_snap][ci]
    noise = 0.15 * np.real(np.diag(Mt)).mean()
    Rt = eps[:, None] * (Mt + noise * np.eye(n)) * eps.conj()[None, :]
    T = int(G["dt"] * E.CHAN_BW)
    Rfull = U.wishart(Rt, T, rng)
    Rtap = U.wishart(Rt, max(int(G["duty"] * T), n + 1), rng)
    W = G["W"][ci]
    out = {}
    # --- tap: StEFCal on the tap covariance, warm start from the applied gains
    g0 = applied["tap"].copy()
    g0[np.abs(g0) < 1e-3] = 1e-3
    Msun = G["Msun"][k_snap][ci]
    if G["free_sun"] and np.abs(Msun).max() > 0:
        g, s_sun, _ = C.stefcal(Rtap, Mm - Msun, W, g0=g0, niter=40, free=[Msun])
        Mfit = Mm - Msun + s_sun[0] * Msun
        out["sun_tap"] = float(s_sun[0])
    else:
        g, _, _ = C.stefcal(Rtap, Mm, W, g0=g0, niter=40)
        Mfit = Mm
    res = (Rtap - np.outer(g, g.conj()) * Mfit) * W
    sig2 = np.sum(np.abs(res) ** 2) / W.sum()
    den = np.sum(np.abs(g[None, :] * Mfit) ** 2 * W, axis=1)
    out["tap"] = (g, sig2 / np.maximum(den, 1e-30))
    # --- image: grid-domain fit to the image made with the applied gains
    ga = applied["image"]
    w = np.where(np.abs(ga) > 1e-3, phi / np.where(np.abs(ga) > 1e-3, ga, 1), 0)
    D = E.scatter(op, (w[:, None] * Rfull * np.conj(w)[None, :])[op.a, op.b])
    live = np.abs(w) > 0
    lv = (phi[op.a] * np.conj(phi[op.b])) * (live[op.a] & live[op.b])
    if G["free_sun"] and np.abs(Msun).max() > 0:
        prob = C.GridProblem(op, lv * (Mm - Msun)[op.a, op.b], G["mask"][ci], free=[lv * Msun[op.a, op.b]])
    else:
        prob = C.GridProblem(op, lv * Mm[op.a, op.b], G["mask"][ci])
    d, _, info = C.gridcal(prob, D, n_outer=G["img_outer"], n_cg=G["img_cg"])
    gi = np.where(live, ga * d, 0)
    out["image"] = (gi, np.where(live, info["var"] * np.abs(ga) ** 2, np.inf))
    # --- EPICal (Beardsley et al. 2017): correlate antennas with the Cyg A pixel
    ge = applied["epical"]
    we = np.where(np.abs(ge) > 1e-3, phi / np.where(np.abs(ge) > 1e-3, ge, 1), 0)
    kb = G["kpix"][k_snap][ci]
    c = np.conj(we * kb)
    K = Rfull @ c - np.diag(Rfull) * c  # self term removed (eq. 15)
    den = Mpt @ (np.conj(phi * kb)) - np.diag(Mpt) * np.conj(phi * kb)
    gest = K / np.where(np.abs(den) > 0, den, 1)
    gam = G["gamma"]
    out["epical"] = ((1 - gam) * ge + gam * gest, None)
    # --- metrics (image fidelity vs ideal gains on the same data)
    if want_img:
        good = np.abs(eps) > 0.1
        wi = np.where(good, phi / np.where(good, eps, 1), 0)
        Di = E.scatter(op, (wi[:, None] * Rfull * np.conj(wi)[None, :])[op.a, op.b])
        ideal = np.real(np.fft.ifft2(Di))
        m = {}
        for meth in ["none", "tap", "image", "epical"]:
            gm = applied[meth]
            wm = np.where(np.abs(gm) > 1e-3, phi / np.where(np.abs(gm) > 1e-3, gm, 1), 0)
            img = np.real(np.fft.ifft2(E.scatter(op, (wm[:, None] * Rfull * np.conj(wm)[None, :])[op.a, op.b])))
            # fit the scale (flux scale is set separately), then normalised residual
            sc = np.sum(img * ideal) / np.sum(img * img)
            m[meth] = (np.linalg.norm(sc * img - ideal) / np.linalg.norm(ideal), sc)
        out["img_metrics"] = m
    return ci, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chan0", type=int, default=3104)
    ap.add_argument("--nchan", type=int, default=16)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--duration", type=float, default=900)
    ap.add_argument("--dt", type=float, default=10.0)
    ap.add_argument("--duty", type=float, default=0.02)
    ap.add_argument("--refresh", type=float, default=60.0)
    ap.add_argument("--start", default="2026-10-16T06:00:00")
    ap.add_argument("--umin", type=float, default=1.0, help="min baseline (wavelengths)")
    ap.add_argument("--gamma", type=float, default=0.35)
    ap.add_argument("--img_outer", type=int, default=2)
    ap.add_argument("--img_cg", type=int, default=15)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--out", default="sim_out.npz")
    ap.add_argument("--sun_factor", type=float, default=1.0, help="true Sun flux / quiet-Sun model")
    ap.add_argument("--free_sun", action="store_true", help="solve the Sun flux in the tap and image solvers")
    a = ap.parse_args()

    cfg = E.EpicConfig()
    chans = a.chan0 + a.step * np.arange(a.nchan)
    nu = chans * E.CHAN_BW
    nu_c = (a.chan0 + 64) * E.CHAN_BW
    t0 = Time(a.start)
    t_grid = np.arange(0, a.duration, a.dt)
    snap_t = np.arange(0, a.duration + a.refresh, a.refresh)
    T0 = time.time()
    ops = [E.build_gridop(cfg, c) for c in chans]
    n = len(ops[0].ant)
    print(f"ops built {time.time() - T0:.0f}s, {n} stands, {len(chans)} channels", flush=True)
    G.update(cfg=cfg, chans=chans, ops=ops, dt=a.dt, duty=a.duty, gamma=a.gamma, free_sun=a.free_sun,
             img_outer=a.img_outer, img_cg=a.img_cg)
    G["phi"] = [F.geometric_phase(c, ops[0].ant) for c in chans]
    G["W"] = [C.baseline_mask(op, n, a.umin / cfg.delta) for op in ops]
    G["mask"] = [C.ml_weights(op, a.umin / cfg.delta) for op in ops]
    ex = F.random_extra_sources()
    G["Mt"], G["Mm"], G["Mpt"], G["kpix"] = [], [], [], []
    for ts in snap_t:
        t = t0 + TimeDelta(ts, format="sec")
        off = {"CygA": (0.10 * np.cos(ts / 400), -0.05), "CasA": (-0.08, 0.10 * np.sin(ts / 300))}
        tr = F.build_snapshot(t, chans, nside=128, extra_sources=ex, flux_scale={"CygA": 1.08, "CasA": 0.93, "TauA": 1.1},
                              offsets_deg=off, fluct_rms=0.1, sun_flux_factor=a.sun_factor)
        mo = F.build_snapshot(t, chans, nside=64)
        G["Mt"].append([tr.matrix(ops[0].ant, i) for i in range(len(chans))])
        G["Mm"].append([mo.matrix(ops[0].ant, i) for i in range(len(chans))])
        # EPICal as published: model of Cyg A and Cas A only
        G["Mpt"].append([mo.matrix(ops[0].ant, i, names=["CygA", "CasA"]) for i in range(len(chans))])
        G.setdefault("Msun", []).append([mo.matrix(ops[0].ant, i, names=["Sun"]) for i in range(len(chans))])
        cyg = [c for c in mo.comps if c.name == "CygA"][0].enu[0]
        G["kpix"].append([pixel_beam(ops[i], i, cyg) for i in range(len(chans))])
        print(f"snapshot t+{ts:.0f}s built ({time.time() - T0:.0f}s)", flush=True)

    rng = np.random.default_rng(42)
    eps, events = true_gains(rng, n, nu, nu_c, t_grid)
    print("jumpers", events["jumpers"], "dead", events["dead"], flush=True)
    methods = ["none", "tap", "image", "epical"]
    applied = {m: np.ones((len(chans), n), complex) for m in methods}
    trackers = {m: C.Tracker(n, nu, nu_c) for m in ["tap", "image"]}
    flagged = {m: np.zeros(n, bool) for m in methods}
    rec = {m: [] for m in methods}
    imgrec = {m: [] for m in methods}
    raw = {m: [] for m in ["tap", "image"]}
    ctx = mp.get_context("fork")
    with ctx.Pool(a.procs) as pool:
        for k, ts in enumerate(t_grid):
            k_snap = int(ts // a.refresh)
            want_img = (k % 3 == 0)
            jobs = [(ci, k, k_snap, {m: applied[m][ci] for m in methods}, eps[k, ci], 1000 * k + ci, want_img)
                    for ci in range(len(chans))]
            res = dict(pool.map(solve_channel, jobs))
            # record errors of what was applied during this interval
            for m in methods:
                errs = [U.gain_error(np.where(np.abs(applied[m][ci]) > 0, applied[m][ci], 1)[np.abs(eps[k, ci]) > 0.1],
                                     eps[k, ci][np.abs(eps[k, ci]) > 0.1]) for ci in range(len(chans))]
                rec[m].append(np.mean(errs, axis=0))
                if want_img:
                    imgrec[m].append(np.mean([res[ci]["img_metrics"][m][0] for ci in range(len(chans))]))
            # trackers
            for m in ["tap", "image"]:
                g = np.array([res[ci][m][0] for ci in range(len(chans))]).T  # [ant, chan]
                var = np.array([res[ci][m][1] for ci in range(len(chans))]).T
                raw[m].append(U.gain_error(g[np.abs(eps[k].T) > 0.1], eps[k].T[np.abs(eps[k].T) > 0.1]))
                trackers[m].update(g, var, a.dt)
                tab = trackers[m].gains().T  # [chan, ant]
                amp = np.abs(tab).mean(axis=0)
                flagged[m] |= amp < 0.15 * np.median(amp)
                if m == "tap":
                    flagged[m] = amp < 0.15 * np.median(amp)  # the tap still sees flagged stands
                tab[:, flagged[m]] = 0
                applied[m] = tab
            ge = np.array([res[ci]["epical"][0] for ci in range(len(chans))])
            applied["epical"] = np.array([C.reference_phase(x) for x in ge])
            line = "  ".join(f"{m} {rec[m][-1][1]:6.2f}deg" for m in methods)
            if "sun_tap" in res[0]:
                line += f"  sun_tap x{np.median([res[ci]['sun_tap'] for ci in range(len(chans))]):.2f}"
            extra = ("  img " + " ".join(f"{m}={imgrec[m][-1]:.3f}" for m in methods)) if want_img else ""
            print(f"t={ts:5.0f}s  {line}{extra}  flagged tap {flagged['tap'].sum()} img {flagged['image'].sum()}  ({time.time() - T0:.0f}s)", flush=True)
    np.savez(a.out, t=t_grid, chans=chans, **{f"err_{m}": np.array(rec[m]) for m in methods},
             **{f"img_{m}": np.array(imgrec[m]) for m in methods}, **{f"raw_{m}": np.array(raw[m]) for m in raw},
             jumpers=events["jumpers"], dead=events["dead"], args=str(vars(a)))


if __name__ == "__main__":
    main()
