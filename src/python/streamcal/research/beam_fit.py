"""Fit a smooth correction to the LSL dipole beam from the fully correlated TBF captures.

    python beam_fit.py OUT.npy CAPTURE.npz [CAPTURE.npz ...]
    env: BF_CHANS (ADP channels, default 1236,1428,1620,1812), BF_POLS (default 0,1),
         BF_NSIDE (diffuse model, default 32), BF_DIFFUSE (sky.DIFFUSE_MODELS entry, default ovrolwa),
         BF_UMIN (wavelengths, default 1), BF_TMIN (earliest UTC hour, default 1.5),
         BF_BASIS (beamcorr.BASES entry, default full), BF_NITER (outer iterations, default 12),
         BF_HALVES (default sum,even,odd), BF_SCALEMAX (stefcal_sky scale bound, default 8),
         BF_LAM (Tikhonov weight on p, relative to the mean diagonal, default 1e-4),
         BF_TIE (sources whose scale is tied to the mean of the free ones, default TauA,VirA),
         BF_SWAP (1: model each polarisation with the other dipole's LSL pattern; a diagnostic),
         BF_SIM (1: replace the data by a simulation with an injected beam error),
         BF_SIM_P (injected p, comma separated, length of the full basis),
         BF_SIM_DIFFUSE (diffuse model of the simulated truth; default BF_DIFFUSE)

Model, per channel and polarisation (beamcorr.py):

    B_true = B_LSL * C(az, el; p),  log C = sum_j p_j Y_j,  C = 1 at the zenith (gauge)
    M_t(p, s) = sum_pix C S_pix a a^H + sum_k s_k C(s_k) S_k a_k a_k^H
    R_t ~ diag(g_t) M_t(p, s) diag(g_t)^H

Unknowns: g_t per snapshot (StEFCal), p (beam), s_k one flux scale per A-team source shared by
all snapshots, and per-snapshot source offsets (refraction) taken from a stefcal_sky pre-fit
against the uncorrected model and held fixed. (p, s) are fitted by Levenberg-Marquardt on the
normal equations pooled over the training snapshots (the
Jacobian is analytic: dM/dp_j = sum_pix Y_j C S a a^H + ..., one batched GEMM per snapshot).
The per-snapshot gains are profiled out of each step (variable projection: the Schur complement of
the gain block of the normal equations, which is cheap in closed form), so p does not crawl while
the gains absorb each change; StEFCal re-solves the gains after every step. Each snapshot is
weighted by 1 / sum |R|^2 W, so the cost is the sum of the fractional residuals. Tau A and Vir A
are seen only at low elevation on short tracks, where their flux is degenerate with the beam;
their scales are tied to the mean of the Cas A and Cyg A scales (BF_TIE).
Weights: baselines >= umin, autos and flagged stands excluded (as stefcal and sky_eval).

Training: the first three quarters of the night; test: the last quarter (sky rotated).
Evaluation per snapshot (half "sum"), with the uncorrected (0) and the corrected (1) beam:
  op0/op1      stefcal_sky with free per-snapshot fluxes and offsets (the operational solve;
               op0 is the sky_eval baseline)
  fix0/fix1    StEFCal only, against the model with the shared source scales from training
               (fix0: scales fitted with p = 0) and the pre-fit offsets
metrics: chi2/dof (median over stands, noise from the autos), fractional residual on the
weighted baselines, and xerr (cross-half model error, noise free).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from astropy.time import Time  # noqa: E402

from streamcal import beamcorr as BC, forward as F, geometry as G, sky as SM, solvers as C  # noqa: E402
import tbf_cal as TC  # noqa: E402

SOL = 299792458.0
env = os.environ.get
want = [int(x) for x in env("BF_CHANS", "1236,1428,1620,1812").split(",")]
pols = [int(x) for x in env("BF_POLS", "0,1").split(",")]
nside = int(env("BF_NSIDE", "32"))
dmodel = env("BF_DIFFUSE", "ovrolwa")
umin = float(env("BF_UMIN", "1"))
tmin = float(env("BF_TMIN", "1.5"))
basis_name = env("BF_BASIS", "full")
terms = BC.BASES[basis_name]
niter = int(env("BF_NITER", "12"))
halves = env("BF_HALVES", "sum,even,odd").split(",")
scalemax = float(env("BF_SCALEMAX", "8"))
lam_p = float(env("BF_LAM", "1e-4"))
sim = int(env("BF_SIM", "0"))
sim_p = np.array([float(x) for x in env("BF_SIM_P", "-0.25,0.15,0.12,-0.08,0.15,0.05").split(",")])
sim_diffuse = env("BF_SIM_DIFFUSE", dmodel)
# diagnostic: model each data polarisation with the other dipole's LSL pattern (pol-label test)
SWAP = int(env("BF_SWAP", "0"))
TIE = [x for x in env("BF_TIE", "TauA,VirA").split(",") if x]
SIM_SCALES = {"CasA": 0.85, "CygA": 0.90, "TauA": 1.0, "VirA": 1.0, "Sun": 1.0}

ant = np.flatnonzero(G.eg.used_stands())
n = len(ant)
r = G.eg.stand_xyz()[ant]
bl = np.linalg.norm(r[:, None, :2] - r[None, :, :2], axis=-1)


def steering(enu, f):
    return np.exp(2j * np.pi * f / SOL * (r @ np.atleast_2d(enu).T))  # [n, P]


def metrics(R, M, g, W, ac, ok, T, Re=None, Ro=None):
    GG = np.outer(g, g.conj())
    res = R - GG * M
    d = np.maximum(ac, 1e-30)
    chi_ant = (np.abs(res) ** 2 / (np.outer(d, d) / T) * W).sum(1) / np.maximum(W.sum(1), 1)
    out = dict(chi2=float(np.median(chi_ant[ok])), resid=float(np.sum(np.abs(res) ** 2 * W) / np.sum(np.abs(R) ** 2 * W)))
    if Re is not None:
        re, ro = Re - GG * M, Ro - GG * M
        out["xerr"] = float(np.real(np.sum(re * ro.conj() * W)) / np.real(np.sum(Re * Ro.conj() * W)))
    return out


class Snap:
    """One capture, channel and polarisation: data, sky components, pre-fit nuisance parameters."""

    def __init__(self, z, ci, f, pol, t, seed=0):
        self.t, self.f, self.pol = t, f, pol
        self.ps = ("XX", "YY")[pol ^ SWAP]
        cc = TC.cable_correction(f, pol)[ant]
        self.R = {}
        for h in ("even", "odd"):
            Rh = z[f"R_{h}"][ci, pol][np.ix_(ant, ant)].astype(np.complex128)
            self.R[h] = cc[:, None] * Rh * np.conj(cc)[None, :]
        self.R["sum"] = 0.5 * (self.R["even"] + self.R["odd"])
        nsamp = int(z["nsamp"])
        self.T = {"even": nsamp, "odd": nsamp, "sum": 2 * nsamp}
        self.ac = np.real(np.diag(self.R["sum"])).copy()
        self.ok = (self.ac > 0.3 * np.median(self.ac)) & (self.ac < 3 * np.median(self.ac))
        W = (bl / (SOL / f) >= umin).astype(float)
        np.fill_diagonal(W, 0)
        self.W = W * np.outer(self.ok, self.ok)
        if sim:
            self._simulate(nsamp, seed)
        snap = F.build_snapshot(t, [f / G.CHAN_BW], pol=self.ps, nside=nside, diffuse_model=dmodel)
        dif = [c for c in snap.comps if c.name == "diffuse"][0]
        self.A = steering(dif.enu, f).astype(np.complex64)
        self.S_d = dif.flux[:, 0].astype(np.float64)
        self.Y_d = BC.basis(dif.enu, terms)
        comps = [c for c in snap.comps if c.name != "diffuse"]
        self.names = [c.name for c in comps]
        self.enu_k = np.array([c.enu[0] for c in comps]).reshape(-1, 3)
        self.S_k = np.array([float(c.flux[0, 0]) for c in comps])
        self.Y_k = BC.basis(self.enu_k, terms) if len(comps) else np.zeros((len(terms), 0))
        self.el_k = np.degrees(np.arcsin(self.enu_k[:, 2]))
        self.az_k = np.degrees(np.arctan2(self.enu_k[:, 0], self.enu_k[:, 1])) % 360
        # pre-fit against the uncorrected model: offsets (held) and the baseline metrics (op0)
        self.op = {}
        self.g0, self.off = self.op_solve(np.zeros(len(terms)), tag="op0")
        self.dirs = [C.shift_direction(e, o) for e, o in zip(self.enu_k, self.off)]
        self.Pk = [np.outer(a, a.conj()) for a in steering(np.array(self.dirs), f).T] if len(comps) else []

    # ---------------------------------------------------------------- model
    def diffuse_matrix(self, w):
        return ((self.A * w.astype(np.float32)) @ self.A.conj().T).astype(np.complex128)

    def model(self, p, s):
        """s: shared scales per source name (dict)."""
        M = self.diffuse_matrix(self.S_d * np.exp(p @ self.Y_d))
        Ck = np.exp(p @ self.Y_k)
        for k, nm in enumerate(self.names):
            M += s[nm] * self.S_k[k] * Ck[k] * self.Pk[k]
        return M

    def jacobian(self, p, s):
        """M and its derivatives: dM/dp_j (list, len J) and dM/ds_k (dict name -> matrix)."""
        J = len(p)
        w = self.S_d * np.exp(p @ self.Y_d)
        Aw = (self.A[None, :, :] * np.vstack([w, w * self.Y_d]).astype(np.float32)[:, None, :]).reshape((J + 1) * n, -1)
        D = (Aw @ self.A.conj().T).reshape(J + 1, n, n).astype(np.complex128)
        M, dp = D[0], [D[j + 1] for j in range(J)]
        Ck = np.exp(p @ self.Y_k)
        ds = {}
        for k, nm in enumerate(self.names):
            Pk = self.S_k[k] * Ck[k] * self.Pk[k]
            M = M + s[nm] * Pk
            for j in range(J):
                dp[j] = dp[j] + s[nm] * self.Y_k[j, k] * Pk
            ds[nm] = Pk
        return M, dp, ds

    # ---------------------------------------------------------------- per-snapshot solves
    def op_solve(self, p, tag):
        """Operational stefcal_sky (free per-snapshot fluxes and offsets) with beam correction p."""
        Cd = np.exp(p @ self.Y_d)
        Ck = np.exp(p @ self.Y_k)
        Md = self.diffuse_matrix(self.S_d * Cd)
        srcs = [(self.enu_k[k], self.S_k[k] * Ck[k]) for k in range(len(self.names))]
        g, sc, off, info = C.stefcal_sky(self.R["sum"], Md, srcs, self.W, r, self.f, n_outer=20,
                                         scale_bounds=[(0.25, scalemax)] * len(srcs))
        M = Md + sum(sc[k] * C.point_matrix(r, self.f, C.shift_direction(s[0], off[k]), s[1]) for k, s in enumerate(srcs))
        m = metrics(self.R["sum"], M, g, self.W, self.ac, self.ok, self.T["sum"], self.R["even"], self.R["odd"])
        m["scales"] = {nm: float(sc[k]) for k, nm in enumerate(self.names)}
        m["frozen"] = [self.names[k] for k in info["frozen"]]
        self.op[tag] = m
        return g, off

    def fixed_solve(self, p, s, g0):
        M = self.model(p, s)
        g, _, _ = C.stefcal(self.R["sum"], M, self.W, g0=g0, niter=80)
        return metrics(self.R["sum"], M, g, self.W, self.ac, self.ok, self.T["sum"], self.R["even"], self.R["odd"])

    # ---------------------------------------------------------------- simulation
    def _simulate(self, nsamp, seed):
        rng = np.random.default_rng(seed)
        snap = F.build_snapshot(self.t, [self.f / G.CHAN_BW], pol=self.ps, nside=nside, diffuse_model=sim_diffuse)
        pt, tterms = sim_p, BC.BASES["full"]  # the truth always uses the full basis
        M = np.zeros((n, n), complex)
        for c in snap.comps:
            a = steering(c.enu, self.f)
            w = c.flux[:, 0] * BC.correction(c.enu, pt, tterms) * SIM_SCALES.get(c.name, 1.0)
            M += (a * w) @ a.conj().T
        msk = self.W > 0
        a0 = np.sqrt(np.median(np.abs(self.R["sum"][msk])) / np.median(np.abs(M[msk])))
        g = a0 * np.exp(rng.normal(0, 0.15, n) + 1j * rng.uniform(-np.pi, np.pi, n))
        V = np.outer(g, g.conj()) * M
        for h in ("even", "odd"):
            sig = np.sqrt(np.outer(self.ac, self.ac) / nsamp / 2)
            N = sig * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
            N = np.triu(N, 1)
            N = N + N.conj().T
            self.R[h] = V + N
            np.fill_diagonal(self.R[h], self.ac)
        self.R["sum"] = 0.5 * (self.R["even"] + self.R["odd"])
        self.sim_g = g


# -------------------------------------------------------------------- pooled fit


def expand(s_free, names):
    """Shared scales for all sources: the free ones, and the tied ones (TIE) at the mean of the free."""
    s = dict(s_free)
    mean = float(np.mean(list(s_free.values()))) if s_free else 1.0
    for nm in names:
        if nm not in s:
            s[nm] = mean
    return s


def snap_weight(sn, h):
    """Each snapshot counts by its fractional residual: weight 1 / sum |R|^2 W."""
    return 1.0 / np.sum(np.abs(sn.R[h]) ** 2 * sn.W)


def solve_gains(snaps, h, gs, p, s):
    out, cost = [], 0.0
    for sn, g in zip(snaps, gs):
        M = sn.model(p, expand(s, sn.names))
        g, _, _ = C.stefcal(sn.R[h], M, sn.W, g0=g, niter=60)
        out.append(g)
        cost += snap_weight(sn, h) * np.sum(np.abs(sn.R[h] - np.outer(g, g.conj()) * M) ** 2 * sn.W)
    return out, cost


def gain_projected_normal(sn, h, g, p, s, free, free_p):
    """Normal equations for theta = (p, s_free) with the per-snapshot gains profiled out.

    Gains are perturbed as g_a (1 + u_a + i v_a): dV_ab = V_ab (u_a + u_b + i (v_a - v_b)), V = G M G^H.
    With Q = |V|^2 W: G_uu = 2 (diag(Q 1) + Q), G_vv = 2 (diag(Q 1) - Q), G_uv = 0;
    B_u = 2 Re sum_b conj(V) X W, B_v = 2 Im sum_b conj(V) X W for a Hermitian column X.
    Returns the Schur complement H - B^T G^+ B and b - B^T G^+ b_g (all data units x weight).
    """
    J = len(p)
    se = expand(s, sn.names)
    M, dp, ds = sn.jacobian(p, se)
    GG = np.outer(g, g.conj())
    V = GG * M
    res = (sn.R[h] - V) * sn.W
    cols = [GG * d * sn.W for d in dp] if free_p else []
    tied = [nm for nm in sn.names if nm not in free]
    for nm in free:
        X = GG * ds[nm] if nm in ds else np.zeros_like(V)
        for tn in tied:  # tied scales follow the mean of the free ones
            X = X + GG * ds[tn] / len(free)
        cols.append(X * sn.W)
    Xm = np.stack([c.ravel() for c in cols], 1)
    H = np.real(Xm.conj().T @ Xm)
    b = np.real(Xm.conj().T @ res.ravel())
    Vc = V.conj()
    Q = np.abs(V) ** 2 * sn.W
    q1 = Q.sum(1)
    Guu = 2 * (np.diag(q1) + Q)
    Gvv = 2 * (np.diag(q1) - Q)
    Z = np.stack([(Vc * c).sum(1) for c in cols], 1)  # [n, npar]
    Bu, Bv = 2 * Z.real, 2 * Z.imag
    zr = (Vc * res).sum(1)
    bu, bv = 2 * zr.real, 2 * zr.imag
    live = q1 > 0
    Gu = Guu[np.ix_(live, live)]
    Gv = Gvv[np.ix_(live, live)] + 1e-9 * np.mean(q1[live]) * np.eye(live.sum())  # overall phase is free
    Su = np.linalg.solve(Gu, np.column_stack([Bu[live], bu[live]]))
    Sv = np.linalg.solve(Gv, np.column_stack([Bv[live], bv[live]]))
    H = H - Bu[live].T @ Su[:, :-1] - Bv[live].T @ Sv[:, :-1]
    b = b - Bu[live].T @ Su[:, -1] - Bv[live].T @ Sv[:, -1]
    return H, b


def fit(snaps, h, src, s_init, free_p=True, log=print):
    """Gauss-Newton / Levenberg-Marquardt on (p, s_free) with the gains profiled out (variable
    projection), the gains re-solved by StEFCal after every step. Returns dict(p, s, H, hist)."""
    J = len(terms)
    free = [nm for nm in src if nm not in TIE]
    K = len(free)
    p = np.zeros(J)
    s = {nm: s_init[nm] for nm in free}
    gs, cost = solve_gains(snaps, h, [sn.g0.copy() for sn in snaps], p, s)
    mu = 1e-3
    hist = [cost]
    npar = (J if free_p else 0) + K
    for it in range(niter):
        T0 = time.time()
        H = np.zeros((npar, npar))
        b = np.zeros(npar)
        for sn, g in zip(snaps, gs):
            Hs, bs = gain_projected_normal(sn, h, g, p, s, free, free_p)
            w = snap_weight(sn, h)
            H += w * Hs
            b += w * bs
        reg = np.zeros(npar)
        if free_p:
            reg[:J] = lam_p * np.mean(np.diag(H)[:J])
        accepted = False
        for _ in range(8):
            A = H + np.diag(mu * np.diag(H) + reg)
            rhs = b - reg * np.concatenate([p if free_p else np.zeros(0), np.zeros(K)])
            delta = np.linalg.solve(A, rhs)
            p2 = p + delta[:J] if free_p else p
            s2 = {nm: max(s[nm] + delta[(J if free_p else 0) + i], 0.05) for i, nm in enumerate(free)}
            gs2, c2 = solve_gains(snaps, h, gs, p2, s2)
            if c2 < cost:
                p, s, gs, accepted = p2, s2, gs2, True
                mu = max(mu / 4, 1e-7)
                break
            mu *= 6
        rel = (cost - c2) / cost if accepted else 0.0
        if accepted:
            cost = c2
        hist.append(cost)
        se = expand(s, src)
        log(f"    [{h}{'' if free_p else ' p=0'}] iter {it}: cost {cost:.6f} {'ok' if accepted else 'rejected'} "
            f"(-{rel:.1e}) mu {mu:.0e}  p [{' '.join(f'{x:+.3f}' for x in p)}]  "
            f"s [{' '.join(f'{nm} {se[nm]:.3f}' for nm in src)}]  ({time.time() - T0:.0f}s)")
        if not accepted or rel < 1e-6:
            break
    return dict(p=p, s=expand(s, src), H=H, hist=hist, free=free)


def main():
    out, files = sys.argv[1], sys.argv[2:]
    zs = []
    for fn in sorted(files):
        z = np.load(fn)
        t = Time(str(z["t"]))
        if t.datetime.hour + t.datetime.minute / 60 >= tmin:
            zs.append((fn, z, t))
    SM.set_epoch(zs[0][2].decimalyear)
    ntest = max(1, len(zs) // 4)
    print(f"ATEAM={SM.ATEAM_MODEL} diffuse {dmodel} nside {nside}; basis {basis_name} {[BC.NAMES[j] for j in terms]}; "
          f"{len(zs)} captures {zs[0][2].isot} .. {zs[-1][2].isot}, last {ntest} held out; "
          f"sim {sim} {('p_true ' + str(sim_p) + ' truth diffuse ' + sim_diffuse) if sim else ''}", flush=True)
    results = {"meta": dict(terms=terms, basis=basis_name, diffuse=dmodel, nside=nside, sim=sim,
                            sim_p=sim_p, sim_diffuse=sim_diffuse, ateam=SM.ATEAM_MODEL)}
    for c in want:
        for p_ in pols:
            T0 = time.time()
            snaps = []
            for i, (fn, z, t) in enumerate(zs):
                ci = int(np.flatnonzero(z["chans"] == c)[0])
                f = float(z["freq"][ci])
                snaps.append(Snap(z, ci, f, p_, t, seed=1000 * c + 10 * i + p_))
            train, test = snaps[:-ntest], snaps[-ntest:]
            src = sorted({nm for sn in train for nm in sn.names})
            s_init = {}
            for nm in src:
                v = [sn.op["op0"]["scales"][nm] for sn in train if nm in sn.names and nm not in sn.op["op0"]["frozen"]]
                s_init[nm] = float(np.median(v)) if v else 1.0
            print(f"chan {c} ({f / 1e6:.2f} MHz) pol {'XY'[p_]}: {len(train)} training, {len(test)} test; sources {src}; "
                  f"pre-fit scales {s_init}; set-up {time.time() - T0:.0f}s", flush=True)
            res = dict(freq=f, src=src, fit={})
            for h in halves:
                res["fit"][h] = fit(train, h, src, s_init, free_p=True, log=lambda x: print(x, flush=True))
            res["fit0"] = fit(train, "sum", src, s_init, free_p=False, log=lambda x: print(x, flush=True))
            pf, sf = res["fit"]["sum"]["p"], res["fit"]["sum"]["s"]
            s0 = res["fit0"]["s"]
            # held-out sources missing from training follow the mean free scale
            allnames = sorted({nm for sn in snaps for nm in sn.names})
            sf = expand({k: sf[k] for k in res["fit"]["sum"]["free"]}, allnames)
            s0 = expand({k: s0[k] for k in res["fit0"]["free"]}, allnames)
            rows = []
            for i, sn in enumerate(snaps):
                sn.op_solve(pf, "op1")
                row = dict(t=sn.t.isot, test=i >= len(train), names=sn.names, el=sn.el_k.tolist(), az=sn.az_k.tolist(),
                           op0=sn.op["op0"], op1=sn.op["op1"],
                           fix0=sn.fixed_solve(np.zeros(len(terms)), s0, sn.g0),
                           fix1=sn.fixed_solve(pf, sf, sn.g0),
                           C_src=np.exp(pf @ sn.Y_k).tolist())
                rows.append(row)
            res["rows"] = rows
            # summary
            te = [x for x in rows if x["test"]]
            tr = [x for x in rows if not x["test"]]

            def med(rr, k, m):
                return np.median([x[k][m] for x in rr])

            print(f"  RESULT chan {c} pol {'XY'[p_]}: p_sum [{' '.join(f'{x:+.3f}' for x in pf)}]  "
                  f"p_even [{' '.join(f'{x:+.3f}' for x in res['fit'].get('even', res['fit']['sum'])['p'])}]  "
                  f"p_odd [{' '.join(f'{x:+.3f}' for x in res['fit'].get('odd', res['fit']['sum'])['p'])}]", flush=True)
            print(f"  RESULT shared scales p=0 {({k: round(v, 3) for k, v in s0.items()})}  with C {({k: round(v, 3) for k, v in sf.items()})}")
            for lab, rr in (("test", te), ("train", tr)):
                print(f"  RESULT {lab}: op chi2 {med(rr, 'op0', 'chi2'):.3f} -> {med(rr, 'op1', 'chi2'):.3f}, "
                      f"resid {med(rr, 'op0', 'resid'):.4f} -> {med(rr, 'op1', 'resid'):.4f}, "
                      f"xerr {med(rr, 'op0', 'xerr'):.4f} -> {med(rr, 'op1', 'xerr'):.4f} | fixed chi2 "
                      f"{med(rr, 'fix0', 'chi2'):.3f} -> {med(rr, 'fix1', 'chi2'):.3f}, resid {med(rr, 'fix0', 'resid'):.4f} -> "
                      f"{med(rr, 'fix1', 'resid'):.4f}, xerr {med(rr, 'fix0', 'xerr'):.4f} -> {med(rr, 'fix1', 'xerr'):.4f}",
                      flush=True)
            print(f"  ({time.time() - T0:.0f}s)", flush=True)
            results[f"{c}_{p_}"] = res
            np.save(out, results, allow_pickle=True)


if __name__ == "__main__":
    main()
