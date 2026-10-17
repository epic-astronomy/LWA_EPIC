"""Kernel-StEFCal on real data: per-stand direction-dependent responses pooled over LST.

    python dd_kernelcal_real.py OUT.npz CAPTURE.npz [CAPTURE.npz ...]
    env: KC_CHANS (ADP channel numbers, comma separated; default 1236,1428,1620,1812)
         KC_POLS (default 0,1), KC_NSIDE (diffuse model, default 32), KC_UMIN (wavelengths,
         default 1), KC_NITER (default 15), KC_LAM (Tikhonov weight toward 0, default 1e-2),
         KC_HOLD (every Nth snapshot held out, default 4), KC_TMIN (earliest UTC hour, default 1.5),
         KC_DIFFUSE (diffuse sky model, a sky.DIFFUSE_MODELS entry; default lfsm),
         KC_SCALEMAX (upper bound on the bright-source flux scales in stefcal_sky, default 4)

Pilot of step 2 in section 8.10 on the Memo 12 TBF captures, correlated in full by
tbf_tcc.py (R_even, R_odd per capture). Model, per channel and polarisation:

    J_a(s, t) = g_a(t) * D_a(s),   D_a(s) = 1 + sum_{j != 0} beta_aj phi_j(s),
    phi_j(s) = exp(2 pi i delta c_j . l(s)),  c_j in {-2..2}^2 (the 5x5 kernel space),

with g_a(t) direction-independent per snapshot and beta pooled over all snapshots (the
instrument is fixed in the station frame; sky-model errors rotate with the sky).
Per snapshot the bright sources' fluxes and refraction offsets come from a
direction-independent stefcal_sky solve first (nuisance parameters, then held fixed).
Alternating: (1) g_t by StEFCal against the model with D applied; (2) beta per stand by
linear least squares over all training snapshots (one batched GEMM per snapshot and
iteration), with a Tikhonov pull toward 0; then the gauge: the mean of beta over stands is
removed (a pattern common to all stands is the station beam, degenerate with the sky).

Solved separately on the even and the odd halves of the samples (signal is common, noise
independent) and on their sum. Held-out snapshots (KC_HOLD): its g is re-solved
with D fixed, and the residual on baselines >= umin is compared with a direction-
independent-only fit (D = 1).
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
n = len(ant)
r = G.eg.stand_xyz()[ant]
bl = np.linalg.norm(r[:, None, :2] - r[None, :, :2], axis=-1)
cells = np.array([(dx, dy) for dy in range(-2, 3) for dx in range(-2, 3)], float)
J0 = 12  # c = (0, 0)
# KC_RMAX: largest |c| of the modes used (2: all 24 direction-dependent modes; 1: the 8
# smoothest, which over-fit sky-model errors less)
JDD = np.array([j for j in range(25) if j != J0 and np.max(np.abs(cells[j])) <= int(os.environ.get("KC_RMAX", "2"))])
delta = 1.0 / (2 * 128 * np.sin(np.radians(1.056) / 2))  # uv cell (wavelengths), epicop.EpicConfig().delta

env = os.environ.get
want = [int(x) for x in env("KC_CHANS", "1236,1428,1620,1812").split(",")]
pols = [int(x) for x in env("KC_POLS", "0,1").split(",")]
nside = int(env("KC_NSIDE", "32"))
umin = float(env("KC_UMIN", "1"))
niter = int(env("KC_NITER", "15"))
lam = float(env("KC_LAM", "1e-2"))
hold = int(env("KC_HOLD", "4"))
tmin = float(env("KC_TMIN", "1.5"))
diffuse_model = env("KC_DIFFUSE", "lfsm")
scalemax = float(env("KC_SCALEMAX", "4"))


def basis(enu):
    """phi_j at directions enu [P, 3] -> [24, P] (the direction-dependent modes)."""
    return np.exp(2j * np.pi * delta * (cells[JDD, 0:1] * enu[None, :, 0] + cells[JDD, 1:2] * enu[None, :, 1]))


class Snap:
    """One capture, one channel and polarisation: data halves, sky components, fixed nuisance fit."""

    def __init__(self, z, ci, f, p, t):
        self.t = t
        cc = TC.cable_correction(f, p)[ant]
        self.R = {}
        for h in ("even", "odd"):
            Rh = z[f"R_{h}"][ci, p][np.ix_(ant, ant)].astype(np.complex128)
            self.R[h] = cc[:, None] * Rh * np.conj(cc)[None, :]
        self.R["sum"] = 0.5 * (self.R["even"] + self.R["odd"])
        ac = np.real(np.diag(self.R["sum"]))
        self.ok = (ac > 0.3 * np.median(ac)) & (ac < 3 * np.median(ac))
        W = (bl / (SOL / f) >= umin).astype(float)
        np.fill_diagonal(W, 0)
        self.W = W * np.outer(self.ok, self.ok)
        snap = F.build_snapshot(t, [f / G.CHAN_BW], pol=("XX", "YY")[p], nside=nside, diffuse_model=diffuse_model)
        Md = snap.matrix(ant, 0, names=["diffuse"])
        srcs = [(c.enu[0], float(c.flux[0, 0]), c.name) for c in snap.comps if c.name != "diffuse"]
        g, sc, off, _ = C.stefcal_sky(self.R["sum"], Md, [(s[0], s[1]) for s in srcs], self.W, r, f, n_outer=20,
                                      scale_bounds=[(0.25, scalemax)] * len(srcs))
        self.g_di = g
        # components with the fitted source fluxes and offsets: enu [P, 3], flux [P]
        dif = [c for c in snap.comps if c.name == "diffuse"][0]
        enu = [dif.enu] + [C.shift_direction(s[0], off[k])[None] for k, s in enumerate(srcs)]
        flx = [dif.flux[:, 0]] + [np.array([s[1] * sc[k]]) for k, s in enumerate(srcs)]
        self.enu = np.concatenate(enu)
        self.S = np.concatenate(flx)
        self.names = [s[2] for s in srcs]
        self.src_enu = {s[2]: C.shift_direction(s[0], off[k]) for k, s in enumerate(srcs)}
        self.A = np.exp(2j * np.pi * f / SOL * (r @ self.enu.T)).astype(np.complex64)  # [n, P]
        self.Phi = basis(self.enu).astype(np.complex64)  # [24, P]

    def model(self, beta):
        """M_ab = sum_s D_a(s) conj(D_b(s)) S(s) a_a(s) conj(a_b(s)) with D = 1 + beta . phi."""
        AD = self.A * (1 + beta.astype(np.complex64) @ self.Phi)  # [n, P]
        return (AD * self.S.astype(np.float32)) @ AD.conj().T


def resid(R, M, g, W):
    res = R - np.outer(g, g.conj()) * M
    return np.sum(np.abs(res) ** 2 * W) / np.sum(np.abs(R) ** 2 * W)


def solve(snaps, h, niter):
    """Alternating g_t (per snapshot) and beta (pooled). Returns beta [n, 24], g list."""
    beta = np.zeros((n, len(JDD)), complex)
    gs = [s.g_di.copy() for s in snaps]
    for it in range(niter):
        T0 = time.time()
        # (1) direction-independent gains per snapshot against the current D
        for k, s in enumerate(snaps):
            gs[k], _, _ = C.stefcal(s.R[h], s.model(beta), s.W, g0=gs[k], niter=40)
        # (2) beta per stand. With g fixed: y_ab = R_ab / (g_a conj g_b) = sum_s D_a phi.. :
        #     y_ab - X0_ab = sum_j beta_aj X_ab,j,  X_ab,j = sum_s phi_j(s) a_a(s) S(s) conj(a_b(s) D_b(s))
        AtA = np.zeros((n, len(JDD), len(JDD)), complex)
        Aty = np.zeros((n, len(JDD)), complex)
        for k, s in enumerate(snaps):
            g = gs[k]
            AD = s.A * (1 + beta.astype(np.complex64) @ s.Phi)  # [n, P]
            Bc = (AD * s.S.astype(np.float32)).conj()  # [n(b), P]
            X0 = s.A @ Bc.T  # [n(a), n(b)]: the j = 0 term (phi = 1)
            # X[a, j, b] in one GEMM: (A[a, s] Phi[j, s]) @ Bc[b, s]^T
            X = ((s.A[:, None, :] * s.Phi[None, :, :]).reshape(n * len(JDD), -1) @ Bc.T).reshape(n, len(JDD), n)
            gg = np.outer(g, g.conj())
            use = (s.W > 0) & (np.abs(gg) > 1e-12)
            y = np.where(use, s.R[h] / np.where(use, gg, 1), 0) - X0
            w = np.where(use, s.W * np.abs(gg) ** 2, 0)  # weight back to data units
            AtA += np.einsum("ajb,akb,ab->ajk", X.conj(), X, w)
            Aty += np.einsum("ajb,ab,ab->aj", X.conj(), y, w)
        reg = lam * np.trace(AtA, axis1=1, axis2=2).real / len(JDD)
        new = np.zeros_like(beta)
        for a in range(n):
            if reg[a] > 0:  # flagged stands have no data
                new[a] = np.linalg.solve(AtA[a] + reg[a] * np.eye(len(JDD)), Aty[a])
        good = np.array([s.ok for s in snaps]).all(0)
        new[~good] = 0
        new[good] -= new[good].mean(0)  # gauge: no pattern common to all stands
        beta = new if it == 0 else 0.5 * (beta + new)
        print(f"    iter {it}: |beta| rms {np.sqrt(np.mean(np.abs(beta[good]) ** 2)):.4f}  ({time.time() - T0:.0f}s)",
              flush=True)
    return beta, gs


def main():
    out, files = sys.argv[1], sys.argv[2:]
    zs = []
    for fn in sorted(files):
        z = np.load(fn)
        t = Time(str(z["t"]))
        if t.datetime.hour + t.datetime.minute / 60 < tmin:
            continue  # the Sun is up
        zs.append((fn, z, t))
    print(f"diffuse model {diffuse_model}, nside {nside}; ", end="")
    print(f"{len(zs)} captures from {zs[0][2].isot} to {zs[-1][2].isot}", flush=True)
    results = {}
    for c in want:
        for p in pols:
            T0 = time.time()
            snaps = []
            for fn, z, t in zs:
                ci = int(np.flatnonzero(z["chans"] == c)[0])
                f = float(z["freq"][ci])
                snaps.append(Snap(z, ci, f, p, t))
            if hold > 0:  # interleaved: held-out snapshots see nearly the training sky
                test_i = [i for i in range(len(snaps)) if i % hold == hold - 1]
            else:  # block: the last quarter of the night, after the sky has rotated
                test_i = list(range(len(snaps) - max(1, len(snaps) // 4), len(snaps)))
            train = [s for i, s in enumerate(snaps) if i not in test_i]
            test = [s for i, s in enumerate(snaps) if i in test_i]
            print(f"chan {c} ({f / 1e6:.2f} MHz) pol {'XY'[p]}: {len(train)} training, {len(test)} held-out snapshots;"
                  f" sky set up in {time.time() - T0:.0f}s", flush=True)
            sol = {}
            for h in ("even", "odd", "sum"):
                print(f"  half {h}", flush=True)
                sol[h], gs = solve(train, h, niter)
            # held-out: residual with D = 1 against D from the 'sum' solution (g re-solved in both)
            ho = []
            for s in test:
                g1, _, _ = C.stefcal(s.R["sum"], s.model(np.zeros_like(sol["sum"])), s.W, g0=s.g_di, niter=60)
                g2, _, _ = C.stefcal(s.R["sum"], s.model(sol["sum"]), s.W, g0=g1, niter=60)
                ho.append((resid(s.R["sum"], s.model(np.zeros_like(sol["sum"])), g1, s.W),
                           resid(s.R["sum"], s.model(sol["sum"]), g2, s.W)))
            ho = np.array(ho)
            # repeatability: phase of D_a toward a grid of directions, even vs odd
            good = np.array([s.ok for s in snaps]).all(0)
            lg = np.linspace(-0.8, 0.8, 9)
            L, M_ = np.meshgrid(lg, lg)
            v = L ** 2 + M_ ** 2 < 0.64
            egrid = np.c_[L[v], M_[v], np.sqrt(1 - L[v] ** 2 - M_[v] ** 2)]
            Pg = basis(egrid)
            De = np.angle(1 + sol["even"] @ Pg)[good]
            Do = np.angle(1 + sol["odd"] @ Pg)[good]
            sig = np.sqrt(max(np.mean(De * Do), 0))
            noi = np.sqrt(np.mean((De - Do) ** 2) / 2)
            corr = np.corrcoef(De.ravel(), Do.ravel())[0, 1]
            # Cyg A / Cas A ratio from the screens at the middle held-out time
            pair = None
            mid = snaps[len(snaps) // 2]
            if "CygA" in mid.src_enu and "CasA" in mid.src_enu:
                e2 = np.array([mid.src_enu["CygA"], mid.src_enu["CasA"]])
                Dp = 1 + sol["sum"] @ basis(e2)
                ratio = Dp[:, 0] / Dp[:, 1]
                pair = float(np.degrees(np.std(np.angle(ratio[good] * np.conj(np.mean(ratio[good]))))))
            print(f"  RESULT chan {c} pol {'XY'[p]}: screen phase signal {np.degrees(sig):.1f} deg rms, "
                  f"even-odd noise {np.degrees(noi):.1f} deg, even/odd corr {corr:+.2f}; held-out residual "
                  f"DI {ho[:, 0].mean():.4f} -> DD {ho[:, 1].mean():.4f} ({100 * (1 - ho[:, 1].mean() / ho[:, 0].mean()):+.1f}%); "
                  f"CygA/CasA phase rms from screens {pair if pair is None else round(pair, 1)} deg; "
                  f"{time.time() - T0:.0f}s", flush=True)
            results[f"{c}_{p}"] = dict(beta=sol, ho=ho, sig=sig, noi=noi, corr=corr, pair=pair, good=good,
                                       times=[s.t.isot for s in snaps], freq=f)
            np.save(out, results, allow_pickle=True)


if __name__ == "__main__":
    main()
