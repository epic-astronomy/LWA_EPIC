"""Are there per-antenna direction-dependent gains at LWA-SV? Measured on TBF data.

    python dd_analyze.py dd_8.npz [dd_16.npz ...]

For each capture, polarisation and clean channel:
  1. direction-independent solve (stefcal_sky: gains, bright-source fluxes and positions)
     on the full covariance (even + odd halves);
  2. per-source gains: for each bright source k, the data minus the diffuse model and the
     other sources (with their current gains) is fitted with that source alone,
     R_k ~ G_k S_k a a^H G_k^H, alternating over k (SAGE-like, 4 rounds), separately for
     the even and the odd half of the samples;
  3. direction-dependent ratio between two sources, d_a = g_a(k1) / g_a(k2), with the
     common phase, the tip/tilt (refraction: a linear phase gradient) and the median
     amplitude removed.
A real effect is the same in both halves; noise is independent. Signal variance is the
real part of the covariance between the halves, noise variance half the variance of
their difference. Also reported: the correlation of d between neighbouring channels
(a property of real, smooth effects, not of noise).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from astropy.time import Time  # noqa: E402

from streamcal import forward as F, geometry as G, solvers as C  # noqa: E402
import tbf_cal as TC  # noqa: E402

SOL = 299792458.0
ant = np.flatnonzero(G.eg.used_stands())
r = G.eg.stand_xyz()[ant]
bl = np.linalg.norm(r[:, None, :2] - r[None, :, :2], axis=-1)


def chi2(R, M, g, W, T):
    d = np.maximum(np.real(np.diag(R)), 1e-30)
    res = np.abs(R - np.outer(g, g.conj()) * M) ** 2 / (np.outer(d, d) / T) * W
    return np.median(res.sum(1) / np.maximum(W.sum(1), 1))


def normalise(d, f):
    """Remove the common phase, the linear phase gradient across the array and the median amplitude."""
    ph = np.angle(d * np.conj(np.sum(d)))
    A = np.c_[np.ones(len(d)), 2 * np.pi * f / SOL * r[:, :2]]
    ok = np.isfinite(ph)
    c = np.linalg.lstsq(A[ok], ph[ok], rcond=None)[0]
    d = d * np.exp(-1j * (A @ c))
    d = d * np.conj(np.sum(d)) / np.abs(np.sum(d))
    return d / np.median(np.abs(d))


def dd_solve(R, Md, srcs, W, f, g_di, sc, off, rounds=4):
    """Per-source gains by alternating rank-1 fits to the source-isolated data."""
    K = len(srcs)
    P = [C.point_matrix(r, f, C.shift_direction(srcs[k][0], off[k]), srcs[k][1] * sc[k]) for k in range(K)]
    gk = [g_di.copy() for _ in range(K)]
    base = np.outer(g_di, g_di.conj()) * Md
    for _ in range(rounds):
        for k in range(K):
            others = sum(np.outer(gk[j], gk[j].conj()) * P[j] for j in range(K) if j != k)
            Rk = R - base - others
            gk[k], _, _ = C.stefcal(Rk, P[k], W, g0=gk[k], niter=60)
    return gk


def main(files):
    rows = []
    for fn in files:
        z = np.load(fn)
        chans, t, T = z["chans"], Time(str(z["t"])), int(z["nsamp"])
        freq = z["freq"] if "freq" in z.files else chans * 25e3
        print(f"== {os.path.basename(fn)} {t.isot}  ({T} samples per half)")
        for p, pol in enumerate(("XX", "YY")):
            snap = F.build_snapshot(t, freq / G.CHAN_BW, pol=pol, nside=64)
            names = [c.name for c in snap.comps if c.name != "diffuse"]
            if len(names) < 2:
                print("   fewer than two bright sources up")
                continue
            dd = {h: [] for h in ("even", "odd")}
            used_f = []
            for ci, f in enumerate(freq):
                halves = {}
                for h in ("even", "odd"):
                    Rh = z[f"R_{h}"][ci, p][np.ix_(ant, ant)].astype(np.complex128)
                    cc = TC.cable_correction(f, p)[ant]
                    halves[h] = cc[:, None] * Rh * np.conj(cc)[None, :]
                R = 0.5 * (halves["even"] + halves["odd"])
                ac = np.real(np.diag(R))
                ok = (ac > 0.3 * np.median(ac)) & (ac < 3 * np.median(ac))
                W = ((bl / (SOL / f)) >= float(os.environ.get("DD_UMIN", "1.0"))).astype(float)
                np.fill_diagonal(W, 0)
                W *= np.outer(ok, ok)
                Md = snap.matrix(ant, ci, names=["diffuse"])
                srcs = [(c.enu[0], float(c.flux[0, ci])) for c in snap.comps if c.name != "diffuse"]
                g, sc, off, _ = C.stefcal_sky(R, Md, srcs, W, r, f, n_outer=20)
                Mfit = Md + sum(sc[k] * C.point_matrix(r, f, C.shift_direction(srcs[k][0], off[k]), srcs[k][1])
                                for k in range(len(srcs)))
                x2 = chi2(R, Mfit, g, W, 2 * T)
                if x2 > float(os.environ.get("DD_MAXCHI", "3")):
                    continue
                used_f.append(f)
                for h in ("even", "odd"):
                    gk = dd_solve(halves[h], Md, srcs, W, f, g, sc, off)
                    # ratio of the two brightest sources
                    order = np.argsort([-s[1] * sc_ for s, sc_ in zip(srcs, sc)])[:2]
                    d = gk[order[0]] / gk[order[1]]
                    d[~ok] = np.nan
                    dd[h].append(normalise(np.where(ok, d, 1.0), f) * np.where(ok, 1, np.nan))
                pair = (names[order[0]], names[order[1]])
            if not used_f:
                continue
            e, o = np.array(dd["even"]), np.array(dd["odd"])  # [chan, ant]
            m = np.isfinite(e) & np.isfinite(o)
            le, lo = np.where(m, np.log(np.where(m, e, 1)), 0), np.where(m, np.log(np.where(m, o, 1)), 0)
            # complex log: real = log amplitude, imag = phase. Signal variance = E[even * odd]
            # (signed, with its standard error); noise variance = E[(even - odd)^2] / 2.
            def stat(a, b):
                prod = (a * b)[np.isfinite(a * b)]
                return prod.mean(), prod.std() / np.sqrt(prod.size), np.sqrt(np.nanmean((a - b) ** 2) / 2)
            vp, ep, noi_ph = stat(np.where(m, le.imag, np.nan), np.where(m, lo.imag, np.nan))
            va, ea, noi_am = stat(np.where(m, le.real, np.nan), np.where(m, lo.real, np.nan))
            avg = 0.5 * (le + lo)
            nb = np.mean([np.corrcoef(avg[i].imag[m[i] & m[i + 1]], avg[i + 1].imag[m[i] & m[i + 1]])[0, 1]
                          for i in range(len(avg) - 1)])
            # band-averaged per antenna: a smooth (frequency-independent) effect accumulates
            ce, co = np.nanmean(np.where(m, le.imag, np.nan), 0), np.nanmean(np.where(m, lo.imag, np.nan), 0)
            vb, eb, noi_avg = stat(ce, co)
            cea, coa = np.nanmean(np.where(m, le.real, np.nan), 0), np.nanmean(np.where(m, lo.real, np.nan), 0)
            vba, eba, noi_avga = stat(cea, coa)
            d2 = np.degrees(1) ** 2
            sig_ph, sig_am, sig_avg = np.sqrt(max(vp, 0)), np.sqrt(max(va, 0)), np.sqrt(max(vb, 0))
            print(f"   {pol} {pair[0]}/{pair[1]}, {len(used_f)} ch | per channel: phase var {vp * d2:+.2f} +- {ep * d2:.2f} deg^2 "
                  f"(noise {np.degrees(noi_ph):.1f} deg), amp var {1e4 * va:+.1f} +- {1e4 * ea:.1f} %^2 (noise {100 * noi_am:.1f}%) | "
                  f"band-avg: phase var {vb * d2:+.2f} +- {eb * d2:.2f} deg^2 (noise {np.degrees(noi_avg):.2f} deg), "
                  f"amp var {1e4 * vba:+.2f} +- {1e4 * eba:.2f} %^2 | nbr corr {nb:+.2f}", flush=True)
            rows.append(dict(file=fn, t=t.isot, pol=pol, pair=pair, sig_ph=sig_ph, noi_ph=noi_ph, sig_am=sig_am,
                             noi_am=noi_am, nb=nb, sig_avg=sig_avg, noi_avg=noi_avg, dd=avg, freq=np.array(used_f),
                             dd_even=le, dd_odd=lo, mask=m))
    np.save(os.environ.get("DD_OUT", "dd_results.npy"), rows, allow_pickle=True)


if __name__ == "__main__":
    main(sys.argv[1:])
