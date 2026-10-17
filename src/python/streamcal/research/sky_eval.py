"""Score a calibration sky model on the fully correlated TBF captures (direction-independent fit).

    STREAMCAL_ATEAM=... STREAMCAL_CATALOG=... python sky_eval.py OUT.npy CAPTURE.npz [...]
    env: EV_CHANS (default 1236,1428,1620,1812), EV_NSIDE (32), EV_DIFFUSE (lfsm | ovrolwa | gsm16), EV_UMIN (1.0), EV_TMIN (1.5 h UTC),
         EV_STRIDE (use every Nth capture, default 2)

Per capture, channel and polarisation: stefcal_sky (gains, A-team fluxes and offsets) against the
forward model, then
  chi2/dof   median over stands of |R - g g* M|^2 / (R_aa R_bb / T), T = 2 nsamp (even + odd)
  resid      sum |R - g g* M|^2 / sum |R|^2 on baselines >= umin
  scale_k    fitted flux scale of each A-team source (1 = the model is right), with its elevation
A better sky model gives lower chi2 and resid, and A-team scales closer to 1 that do not drift
with elevation (beam errors also make them drift).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from astropy.time import Time  # noqa: E402

from streamcal import forward as F, geometry as G, sky as SM, solvers as C  # noqa: E402
import tbf_cal as TC  # noqa: E402

SOL = 299792458.0
env = os.environ.get
chans = [int(x) for x in env("EV_CHANS", "1236,1428,1620,1812").split(",")]
nside = int(env("EV_NSIDE", "32"))
umin = float(env("EV_UMIN", "1.0"))
tmin = float(env("EV_TMIN", "1.5"))
stride = int(env("EV_STRIDE", "2"))
dmodel = env("EV_DIFFUSE", "lfsm")
ant = np.flatnonzero(G.eg.used_stands())
r = G.eg.stand_xyz()[ant]
bl = np.linalg.norm(r[:, None, :2] - r[None, :, :2], axis=-1)


def main():
    out, files = sys.argv[1], sorted(sys.argv[2:])
    caps = []
    for fn in files:
        z = np.load(fn)
        t = Time(str(z["t"]))
        if t.datetime.hour + t.datetime.minute / 60 >= tmin:
            caps.append((z, t))
    caps = caps[::stride]
    SM.set_epoch(caps[0][1].decimalyear)
    print(f"model: ATEAM={SM.ATEAM_MODEL} DIFFUSE={dmodel} CATALOG={SM.CATALOG or 'none'} (min {SM.CATALOG_MIN} Jy); "
          f"{len(caps)} captures, channels {chans}", flush=True)
    rows = []
    for z, t in caps:
        for c in chans:
            ci = int(np.flatnonzero(z["chans"] == c)[0])
            f = float(z["freq"][ci])
            T = 2 * int(z["nsamp"])
            for p in (0, 1):
                cc = TC.cable_correction(f, p)[ant]
                R = 0.5 * (z["R_even"][ci, p] + z["R_odd"][ci, p])[np.ix_(ant, ant)].astype(np.complex128)
                R = cc[:, None] * R * np.conj(cc)[None, :]
                ac = np.real(np.diag(R))
                ok = (ac > 0.3 * np.median(ac)) & (ac < 3 * np.median(ac))
                W = (bl / (SOL / f) >= umin).astype(float)
                np.fill_diagonal(W, 0)
                W *= np.outer(ok, ok)
                snap = F.build_snapshot(t, [f / G.CHAN_BW], pol=("XX", "YY")[p], nside=nside, diffuse_model=dmodel)
                Md = snap.matrix(ant, 0, names=["diffuse"])
                srcs = [(cp.enu[0], float(cp.flux[0, 0]), cp.name) for cp in snap.comps if cp.name != "diffuse"]
                g, sc, off, _ = C.stefcal_sky(R, Md, [(s[0], s[1]) for s in srcs], W, r, f, n_outer=20)
                M = Md + sum(sc[k] * C.point_matrix(r, f, C.shift_direction(s[0], off[k]), s[1])
                             for k, s in enumerate(srcs))
                res = R - np.outer(g, g.conj()) * M
                d = np.maximum(ac, 1e-30)
                chi_ant = (np.abs(res) ** 2 / (np.outer(d, d) / T) * W).sum(1) / np.maximum(W.sum(1), 1)
                rows.append(dict(t=t.isot, chan=c, pol=p, chi2=float(np.median(chi_ant[ok])),
                                 resid=float(np.sum(np.abs(res) ** 2 * W) / np.sum(np.abs(R) ** 2 * W)),
                                 scales={s[2]: float(sc[k]) for k, s in enumerate(srcs)},
                                 elev={s[2]: float(np.degrees(np.arcsin(s[0][2]))) for s in srcs}))
        print(f"  {t.isot} done", flush=True)
    np.save(out, rows, allow_pickle=True)
    # summary
    print(f"{'chan':>5} {'pol':>3} {'chi2/dof':>9} {'resid':>7}   A-team scale median [min-max]")
    for c in chans:
        for p in (0, 1):
            rr = [x for x in rows if x["chan"] == c and x["pol"] == p]
            sc = {}
            for x in rr:
                for k, v in x["scales"].items():
                    sc.setdefault(k, []).append(v)
            s = "  ".join(f"{k} {np.median(v):.2f} [{min(v):.2f}-{max(v):.2f}]" for k, v in sorted(sc.items()) if k != "Sun")
            print(f"{c:>5} {'XY'[p]:>3} {np.median([x['chi2'] for x in rr]):9.2f} {np.median([x['resid'] for x in rr]):7.4f}   {s}")
    allc = [x["chi2"] for x in rows]
    allr = [x["resid"] for x in rows]
    print(f"ALL: chi2/dof median {np.median(allc):.2f}, resid median {np.median(allr):.4f}")


if __name__ == "__main__":
    main()
