"""Real-data check of the forward model + StEFCal on LWA-SV TBF captures (EPIC Memo 12 data).

For each file: correlate T samples at a set of channels, apply the LSL cable
model (as gen_phases_lwasv does), solve per channel and polarisation with
StEFCal against the forward-modelled sky, and report fit quality, delay
coherence across channels, and the solutions (saved for stability checks).
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streamcal import sky as skymodel, epicop, forward, solvers as calib  # noqa: E402,F401
sys.modules.update({"skymodel": skymodel, "epicop": epicop, "forward": forward, "calib": calib})


import argparse
import time

import numpy as np
from astropy.time import Time
from lsl.common.stations import lwasv
from lsl.reader import tbf

import forward as F
import calib as C
import epicop as E

SOL = 299792458.0


def load_tbf(path, chans_want, nsamp, skip=0):
    """X[chan, stand, pol, t] for the wanted ADP channel numbers (25 kHz channels)."""
    chans_want = np.asarray(chans_want)
    blocks = {}
    with open(path, "rb") as f:
        fr = tbf.read_frame(f)
        f.seek(0)
        t0 = None
        times = {}
        while True:
            try:
                fr = tbf.read_frame(f)
            except Exception:
                break
            tt = fr.payload.timetag
            if t0 is None:
                t0 = tt
                ttime = fr.time
            k = (tt - t0) // 7840  # 196e6/25e3 ticks per sample
            if k < skip:
                continue
            k -= skip
            if k >= nsamp:
                if all(len(v) >= nsamp for v in times.values()) and len(times) > 0:
                    break
                continue
            fc = fr.header.first_chan
            sel = (chans_want >= fc) & (chans_want < fc + 12)
            if not sel.any():
                continue
            d = fr.payload.data  # (12, 256, 2)
            for c in chans_want[sel]:
                blocks.setdefault(c, np.zeros((256, 2, nsamp), np.complex64))[:, :, k] = d[c - fc]
                times.setdefault(c, set()).add(k)
    X = np.stack([blocks[c] for c in chans_want])
    nvalid = np.array([len(times[c]) for c in chans_want])
    return X, nvalid, Time(ttime.datetime, scale="utc")


def cable_correction(freq_hz, pol):
    ants = lwasv.antennas[pol::2]
    return np.array([np.exp(2j * np.pi * freq_hz * a.cable.delay(freq_hz)) / np.sqrt(a.cable.gain(freq_hz)) for a in ants])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--nsamp", type=int, default=10000)
    ap.add_argument("--chans", default="1140:1900:48")
    ap.add_argument("--umin", type=float, default=1.0)
    ap.add_argument("--out", default="tbf_solutions.npz")
    a = ap.parse_args()
    c0, c1, cs = (int(x) for x in a.chans.split(":"))
    chans = np.arange(c0, c1, cs)
    freq = chans * 25e3
    used = E.eg.used_stands()
    ant = np.flatnonzero(used)
    xyz = E.eg.stand_xyz()[ant]
    bl = np.linalg.norm(xyz[:, None, :2] - xyz[None, :, :2], axis=-1)
    results = []
    for path in a.files:
        T0 = time.time()
        X, nvalid, t = load_tbf(path, chans, a.nsamp)
        print(f"{path.split('/')[-1]}  {t.isot}  loaded {X.shape} valid {nvalid.min()}..{nvalid.max()} ({time.time() - T0:.0f}s)", flush=True)
        snapX = F.build_snapshot(t, freq / E.CHAN_BW, pol="XX", nside=64)
        snapY = F.build_snapshot(t, freq / E.CHAN_BW, pol="YY", nside=64)
        sol = np.zeros((len(chans), 2, len(ant)), complex)
        stats = np.zeros((len(chans), 2, 4))
        for ci, f in enumerate(freq):
            lam = SOL / f
            W = ((bl / lam) >= a.umin).astype(float)
            np.fill_diagonal(W, 0)
            for p, snap in enumerate([snapX, snapY]):
                x = X[ci, ant, p, :]
                good = np.abs(x).sum(1) > 0
                R = (x @ x.conj().T) / x.shape[1]
                cc = cable_correction(f, p)[ant]
                Rc = cc[:, None] * R * np.conj(cc)[None, :]
                M = snap.matrix(ant, ci)
                Wg = W * np.outer(good, good)
                out = []
                for conv, Rd in (("as-is", Rc), ("conj", np.conj(Rc))):
                    g, s, it = C.stefcal(Rd, M, Wg, niter=200)
                    res = (Rd - np.outer(g, g.conj()) * M) * Wg
                    out.append((np.linalg.norm(res) / np.linalg.norm(Rd * Wg), g, it, conv))
                best = min(out, key=lambda z: z[0])
                # model/data correlation before calibration (cable model only)
                rho0 = np.abs(np.vdot(M * Wg, Rc * Wg)) / np.linalg.norm(M * Wg) / np.linalg.norm(Rc * Wg)
                sol[ci, p] = best[1]
                stats[ci, p] = [out[0][0], out[1][0], rho0, best[2]]
        # delay coherence: fit phase slope per antenna across channels
        ph = np.unwrap(np.angle(sol / sol[len(chans) // 2]), axis=0)
        A = np.c_[np.ones_like(freq), 2 * np.pi * (freq - freq.mean())]
        coef, *_ = np.linalg.lstsq(A, ph.reshape(len(chans), -1), rcond=None)
        resid = ph.reshape(len(chans), -1) - A @ coef
        tau_ns = coef[1].reshape(2, -1) * 1e9
        print(f"   rel. residual |R-GMG^H|/|R|: as-is {np.median(stats[:, :, 0]):.3f}  conj {np.median(stats[:, :, 1]):.3f};"
              f"  data-model corr before cal {np.median(stats[:, :, 2]):.3f};  iters {np.median(stats[:, :, 3]):.0f}")
        print(f"   residual delays (ns) X rms {tau_ns[0].std():.2f}  Y rms {tau_ns[1].std():.2f};  phase rms about delay fit {np.degrees(resid.std()):.1f} deg;"
              f"  |g| spread {np.std(np.abs(sol)) / np.mean(np.abs(sol)):.2f}", flush=True)
        results.append(dict(path=path, t=t.isot, sol=sol, stats=stats, tau=tau_ns))
    np.savez(a.out, chans=chans, ant=ant, t=[r["t"] for r in results], sol=np.array([r["sol"] for r in results]),
             stats=np.array([r["stats"] for r in results]), tau=np.array([r["tau"] for r in results]))


if __name__ == "__main__":
    main()
