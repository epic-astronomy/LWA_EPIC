"""Direction-independent calibration quality against different diffuse sky models.

    python diffuse_eval.py OUT.npy CAPTURE.npz [CAPTURE.npz ...]
    env: DE_MODELS (sky.DIFFUSE_MODELS entries, comma separated; default lfsm,ovrolwa)
         DE_NSIDE (default 64), DE_CHANS (ADP channels; default 1236,1428,1620,1812),
         DE_POLS (default 0,1), DE_UMINS (baseline cuts in wavelengths; default 1,3,6),
         DE_TMIN (earliest UTC hour, default 1.5: the Sun has set), DE_SCALEMAX (upper
         guard-rail bound on the bright-source flux scales in stefcal_sky, default 4; Cas A
         fits at 3.5-3.8 at 30.9 MHz against the Baars spectrum, so 4 freezes it at random)

Per capture, polarisation, channel, diffuse model and baseline cut: the stefcal_sky
solve of dd_analyze.py (cable correction from tbf_cal, gains plus bright-source fluxes
and positions, the diffuse sky fixed) on the full covariance (even + odd halves), then
  chi2   median over stands of the mean over baselines of |R - g M g^H|^2 / (R_aa R_bb / T),
         T = 2 nsamp (the sum of the halves), on baselines >= umin;
  frac   sum |R - g M g^H|^2 / sum |R|^2 on the same baselines;
  xerr   the sky-model error alone: Re sum res_even conj(res_odd) / Re sum R_even conj(R_odd),
         with res_h = R_h - g M g^H for each half and g, M from the joint solve. The noise
         of the two halves is independent, so it drops out of the cross product and xerr
         is the fraction of the correlated sky power the model leaves unexplained;
  dscale the least-squares amplitude of the diffuse model with g and the point sources held
         at the solution (1 = the model's diffuse brightness is right relative to the A-team);
  scale  the fitted flux scale of each bright source, with its elevation.
Captures can be the full tbf_tcc.py files or channel subsets from tbf_subset.py.
"""
import os
import sys
import time

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
env = os.environ.get
models = env("DE_MODELS", "lfsm,ovrolwa").split(",")
nside = int(env("DE_NSIDE", "64"))
want = [int(x) for x in env("DE_CHANS", "1236,1428,1620,1812").split(",")]
pols = [int(x) for x in env("DE_POLS", "0,1").split(",")]
umins = [float(x) for x in env("DE_UMINS", "1,3,6").split(",")]
tmin = float(env("DE_TMIN", "1.5"))
scalemax = float(env("DE_SCALEMAX", "4"))


def chi2(R, M, g, W, T):
    d = np.maximum(np.real(np.diag(R)), 1e-30)
    res = np.abs(R - np.outer(g, g.conj()) * M) ** 2 / (np.outer(d, d) / T) * W
    return float(np.median(res.sum(1)[W.sum(1) > 0] / W.sum(1)[W.sum(1) > 0]))


def main(out, files):
    rows = []
    for fn in sorted(files):
        z = np.load(fn)
        t = Time(str(z["t"]))
        if t.datetime.hour + t.datetime.minute / 60 < tmin:
            continue
        T = 2 * int(z["nsamp"])
        cis = [int(np.flatnonzero(z["chans"] == c)[0]) for c in want]
        freq = np.array([float(z["freq"][ci]) for ci in cis])
        Re, Ro = z["R_even"], z["R_odd"]
        T0 = time.time()
        for p in pols:
            for model in models:
                snap = F.build_snapshot(t, freq / G.CHAN_BW, pol=("XX", "YY")[p], nside=nside, diffuse_model=model)
                for j, (c, ci, f) in enumerate(zip(want, cis, freq)):
                    cc = TC.cable_correction(f, p)[ant]
                    Rh = [cc[:, None] * X[ci, p][np.ix_(ant, ant)].astype(np.complex128) * np.conj(cc)[None, :]
                          for X in (Re, Ro)]
                    R = 0.5 * (Rh[0] + Rh[1])
                    ac = np.real(np.diag(R))
                    ok = (ac > 0.3 * np.median(ac)) & (ac < 3 * np.median(ac))
                    Md = snap.matrix(ant, j, names=["diffuse"])
                    comps = [x for x in snap.comps if x.name != "diffuse"]
                    srcs = [(x.enu[0], float(x.flux[0, j])) for x in comps]
                    for umin in umins:
                        W = (bl / (SOL / f) >= umin).astype(float)
                        np.fill_diagonal(W, 0)
                        W *= np.outer(ok, ok)
                        g, sc, off, info = C.stefcal_sky(R, Md, srcs, W, r, f, n_outer=20,
                                                         scale_bounds=[(0.25, scalemax)] * len(srcs))
                        M = Md + sum(sc[k] * C.point_matrix(r, f, C.shift_direction(srcs[k][0], off[k]), srcs[k][1])
                                     for k in range(len(srcs)))
                        GG = np.outer(g, g.conj())
                        res = np.abs(R - GG * M) ** 2
                        xe = np.sum((Rh[0] - GG * M) * np.conj(Rh[1] - GG * M) * W).real
                        xs = np.sum(Rh[0] * np.conj(Rh[1]) * W).real
                        Mp = M - Md
                        a = (GG * Md * W).ravel()
                        dscale = float(np.vdot(a, ((R - GG * Mp) * W).ravel()).real / np.vdot(a, a).real)
                        rows.append(dict(t=t.isot, hour=t.datetime.hour + t.datetime.minute / 60, pol=p, chan=c, freq=f,
                                         model=model, umin=umin, chi2=chi2(R, M, g, W, T),
                                         frac=float(np.sum(res * W) / np.sum(np.abs(R) ** 2 * W)),
                                         xerr=float(xe / xs), dscale=dscale,
                                         names=[x.name for x in comps],
                                         alt=[float(np.degrees(np.arcsin(x.enu[0, 2]))) for x in comps],
                                         scale=sc.tolist(), off_deg=np.degrees(np.hypot(off[:, 0], off[:, 1])).tolist(),
                                         frozen=info["frozen"], app=[s[1] for s in srcs]))
        print(f"{os.path.basename(fn)} {t.isot} done in {time.time() - T0:.0f}s", flush=True)
        np.save(out, rows, allow_pickle=True)
    summarise(rows)


def summarise(rows):
    import collections

    key = collections.defaultdict(list)
    for x in rows:
        key[(x["chan"], x["pol"], x["umin"], x["model"])].append(x)
    print("\nchan  pol umin model     chi2/dof  frac    xerr    dscale  (medians)  n")
    for k in sorted(key):
        v = key[k]
        med = {m: np.median([x.get(m, np.nan) for x in v]) for m in ("chi2", "frac", "xerr", "dscale")}
        print(f"{k[0]} {'XY'[k[1]]:>3} {k[2]:4.0f} {k[3]:9s} {med['chi2']:6.3f}  {med['frac']:.4f}  {med['xerr']:.4f}"
              f"  {med['dscale']:.3f}   {len(v)}")
    print("\nA-team flux scales (umin 1, unfrozen solves): median, robust scatter (1.4826 MAD) and the slope")
    print("d(scale)/d(sin alt) after 3-MAD clipping; n = solves used, fz = frozen by the guard rails")
    for k in sorted(key):
        if k[2] != 1:
            continue
        per = collections.defaultdict(list)
        for x in key[k]:
            for i, (nme, a, s) in enumerate(zip(x["names"], x["alt"], x["scale"])):
                per[nme].append((a, s, i in x["frozen"]))
        txt = []
        for nme in ("CygA", "CasA", "TauA", "VirA"):
            if nme not in per:
                continue
            v = np.array(per[nme], float)
            nf = int(v[:, 2].sum())
            v = v[v[:, 2] == 0]
            if len(v) < 3:
                txt.append(f"{nme} n={len(v)} fz={nf}")
                continue
            med = np.median(v[:, 1])
            mad = 1.4826 * np.median(np.abs(v[:, 1] - med))
            keep = np.abs(v[:, 1] - med) <= 3 * max(mad, 1e-3)
            sl = np.polyfit(np.sin(np.radians(v[keep, 0])), v[keep, 1], 1)[0]
            txt.append(f"{nme} {med:.2f}+-{mad:.2f} slope {sl:+.2f} n={keep.sum()} fz={nf}")
        print(f"{k[0]} {'XY'[k[1]]} {k[3]:8s} " + "; ".join(txt))
    models = sorted({x["model"] for x in rows})
    if "lfsm" in models:
        print("\nChange against lfsm (medians over snapshots; negative = smaller error):")
        print("chan pol umin model    excess chi2 (chi2-1)   xerr")
        for k in sorted(key):
            if k[3] == "lfsm":
                continue
            ref = key[(k[0], k[1], k[2], "lfsm")]
            e0 = np.median([x["chi2"] for x in ref]) - 1
            e1 = np.median([x["chi2"] for x in key[k]]) - 1
            x0 = np.median([x["xerr"] for x in ref])
            x1 = np.median([x["xerr"] for x in key[k]])
            # paired per-snapshot change of xerr (median), robust to the occasional failed solve
            pr = {(x["t"]): x["xerr"] for x in ref}
            pc = np.median([x["xerr"] / pr[x["t"]] - 1 for x in key[k] if x["t"] in pr and pr[x["t"]] > 0])
            print(f"{k[0]} {'XY'[k[1]]:>3} {k[2]:4.0f} {k[3]:8s} {e0:.3f} -> {e1:.3f} ({100 * (e1 / e0 - 1):+4.0f}%)"
                  f"   {x0:.4f} -> {x1:.4f} ({100 * (x1 / x0 - 1):+4.0f}%, paired {100 * pc:+4.0f}%)")


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summarise(list(np.load(sys.argv[2], allow_pickle=True)))
    else:
        main(sys.argv[1], sys.argv[2:])
