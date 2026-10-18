"""Kernel-StEFCal: solve each stand's direction-dependent response in the space a 5x5 kernel spans.

Model. Stand a's voltage response toward direction s is
    J_a(s) = sum_j beta_aj phi_j(s),   phi_j(s) = exp(2 pi i delta c_j . l(s)),  c_j in {-2..2}^2,
the 25 low-order Fourier modes a 5x5 gridding kernel can represent. beta_a0 at c = 0 is the
direction-independent gain. The covariance is
    R_ab = sum_s J_a(s) J_b(s)^* S(s) a_a(s) a_b(s)^*,
linear in beta_a for fixed beta_b: R_ab = sum_j beta_aj X_ab,j with
    X_a[j, b] = sum_s phi_j(s) a_a(s) S(s) conj(a_b(s) J_b(s)),
so each stand is solved by linear least squares (25 unknowns, ~228 baselines per snapshot
and channel), alternating over stands with StEFCal-style averaging and a Tikhonov pull
toward the previous iterate. Pooling snapshots at different LST separates the instrument
(fixed in the station frame) from sky-model errors (which rotate with the sky).

Test. Truth: per-stand smooth phase screens (correlation length THETA, scaled to 20 deg rms
of arg J(CygA)/J(CasA), as measured), DI gains (15% amplitude, random phase), sky = LFSM
nside 64 x (1 + 10% small-scale structure) + 25 faint sources + A-team; model = LFSM nside 32
+ A-team. Fit on NFIT snapshots over several hours of LST, evaluate on a held-out LST:
EPIC image error against the ideal image, for (a) DI calibration and (b) kernels fitted to
k_a(s) / J_a(s) from the solved screens.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
from astropy.time import Time, TimeDelta  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

from streamcal import epicop as E, forward as F, solvers as C  # noqa: E402

SOL = 299792458.0
cfg = E.EpicConfig()
chan = int(os.environ.get("KC_CHAN", "1440"))
f = chan * E.CHAN_BW
THETA = float(os.environ.get("KC_THETA", "20"))
rng = np.random.default_rng(11)
op0 = E.build_gridop(cfg, chan)
ant = op0.ant
n = len(ant)
r = F.stand_enu()[ant]
phi = F.geometric_phase(chan, ant)
cells = np.array([(dx, dy) for dy in range(-2, 3) for dx in range(-2, 3)], float)
ng = 96
lg = np.linspace(-1, 1, ng)


def basis(enu):
    """phi_j at directions enu [P, 3] -> [25, P]."""
    return np.exp(2j * np.pi * cfg.delta * (cells[:, 0:1] * enu[None, :, 0] + cells[:, 1:2] * enu[None, :, 1]))


def make_truth():
    s = np.radians(THETA)
    k = np.fft.fftfreq(ng, d=lg[1] - lg[0])
    KX, KY = np.meshgrid(k, k)
    filt = np.exp(-2 * (np.pi * s) ** 2 * (KX ** 2 + KY ** 2))
    fld = np.array([np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(ng, ng))) * filt)) for _ in range(n)])
    fld /= fld.std()
    interp = [RegularGridInterpolator((lg, lg), fa) for fa in fld]
    g0 = np.exp(rng.normal(0, 0.15, n) + 1j * rng.uniform(-np.pi, np.pi, n))

    def J(enu, scale):
        pts = np.c_[np.clip(enu[:, 1], -1, 1), np.clip(enu[:, 0], -1, 1)]
        return g0[:, None] * np.exp(1j * scale * np.array([it(pts) for it in interp]))
    return J


def components(snap):
    """(enu [P, 3], flux [P]) of all sky components of a one-channel snapshot."""
    e = np.concatenate([c.enu for c in snap.comps])
    s = np.concatenate([c.flux[:, 0] for c in snap.comps])
    return e, s


def vis_from(enu, S, Jv):
    A = np.exp(2j * np.pi * f / SOL * (r @ enu.T)) * Jv
    return (A * S) @ A.conj().T


def solve(snaps_R, snaps_model, beta0, niter=30, lam=1e-3, umin=1.0):
    """Alternating least squares for beta [n, 25] pooled over snapshots."""
    beta = beta0.copy()
    W = (np.linalg.norm(r[:, None, :2] - r[None, :, :2], axis=-1) >= umin * SOL / f).astype(float)
    np.fill_diagonal(W, 0)
    pre = []
    for enu, S in snaps_model:
        Ph = basis(enu)  # [25, P]
        Aa = np.exp(2j * np.pi * f / SOL * (r @ enu.T))  # [n, P]
        pre.append((Ph, Aa, S))
    for it in range(niter):
        new = np.empty_like(beta)
        Jm = [beta @ Ph * Aa for Ph, Aa, S in pre]  # J_b(s) a_b(s) [n, P]
        for a in range(n):
            rows_A, rows_y = [], []
            for (Ph, Aa, S), Jb, R in zip(pre, Jm, snaps_R):
                X = (Ph * Aa[a] * S) @ Jb.conj().T  # [25, n]
                m = W[a] > 0
                rows_A.append(X[:, m].T)
                rows_y.append(R[a, m])
            A = np.vstack(rows_A)
            y = np.concatenate(rows_y)
            reg = lam * np.linalg.norm(A) ** 2 / A.shape[1]
            new[a] = np.linalg.solve(A.conj().T @ A + reg * np.eye(25), A.conj().T @ y + reg * beta[a])
        beta = new if it % 2 == 0 else 0.5 * (new + beta)
    return beta


def image(op, R, w):
    return E.image_from_vis(op, w[:, None] * R * np.conj(w)[None, :], with_autos=False)


def main():
    J = make_truth()
    t0 = Time("2021-04-04T08:00:00")
    lst_h = [0, 1.5, 3, 4.5, 6]
    held = 3.75
    ex = F.random_extra_sources()
    truth, model = {}, {}
    for h in lst_h + [held]:
        t = t0 + TimeDelta(h * 3600, format="sec")
        truth[h] = components(F.build_snapshot(t, [chan], nside=64, extra_sources=ex, fluct_rms=0.1))
        model[h] = components(F.build_snapshot(t, [chan], nside=32))
    # scale the screens to 20 deg rms of the Cyg A / Cas A phase ratio at the held-out time
    snap_h = F.build_snapshot(t0 + TimeDelta(held * 3600, format="sec"), [chan], nside=0)
    pos = {c.name: c.enu[0] for c in snap_h.comps}
    u = np.angle(J(np.array([pos["CygA"], pos["CasA"]]), 1.0))
    scale = np.radians(20) / np.std(np.angle(np.exp(1j * (u[:, 0] - u[:, 1]))))
    Jt = lambda enu: J(enu, scale)  # noqa: E731
    R = {h: vis_from(*truth[h], Jt(truth[h][0])) for h in truth}
    print(f"{f / 1e6:.1f} MHz, screen scale {THETA:.0f} deg; fitting on LST offsets {lst_h} h, held out {held} h", flush=True)
    # start from the DI solution of the first snapshot
    W = np.ones((n, n)) - np.eye(n)
    g, _, _ = C.stefcal(R[lst_h[0]], vis_from(*model[lst_h[0]], 1.0), W, niter=200)
    beta0 = np.zeros((n, 25), complex)
    beta0[:, 12] = g  # c = (0, 0) is index 12
    beta = solve([R[h] for h in lst_h], [model[h] for h in lst_h], beta0, niter=int(os.environ.get('KC_NITER', '30')))
    # evaluate on the held-out snapshot
    enu_h, S_h = truth[held]
    ideal = image(op0, vis_from(enu_h, S_h, 1.0), phi)
    P1 = C.point_matrix(r, f, pos["CygA"], 1.0)
    jy = 1.0 / image(op0, P1, phi).max()
    kk = np.fft.fftfreq(op0.N) * op0.N
    lp = kk / (op0.N * cfg.delta)
    LL, MM = np.meshgrid(lp, lp, indexing="ij")
    vis = LL ** 2 + MM ** 2 < 0.95
    g_di, _, _ = C.stefcal(R[held], vis_from(*model[held], 1.0), W, niter=200)
    img_a = image(op0, R[held], phi / g_di)
    # fitted screens -> kernels: k'_a(s) = k_a(s) / J_a(s)
    kc, xp, yp, posg, used = E.kernels(cfg, chan)
    Lg, Mg = np.meshgrid(lg, lg)
    vv = Lg ** 2 + Mg ** 2 < 0.98
    egrid = np.c_[Lg[vv], Mg[vv], np.sqrt(np.clip(1 - Lg[vv] ** 2 - Mg[vv] ** 2, 0, 1))]
    Jfit = beta @ basis(egrid)  # [n, P]
    wts = np.sqrt(np.sqrt(egrid[:, 2]))
    kc2 = kc.copy()
    for i, a in enumerate(ant):
        B = np.exp(-2j * np.pi * cfg.delta * (np.outer(egrid[:, 0], xp[a] - posg[a, 0]) + np.outer(egrid[:, 1], yp[a] - posg[a, 1])))
        target = (B @ kc[a]) / Jfit[i]
        kc2[a] = np.linalg.lstsq(B * wts[:, None], target * wts, rcond=None)[0]
    op_b = E.build_gridop(cfg, chan, kc_override=kc2)
    img_b = image(op_b, R[held], phi)
    # truth screens through the same kernel fit (upper bound of what kernels can do)
    Jtrue = Jt(egrid)
    kc3 = kc.copy()
    for i, a in enumerate(ant):
        B = np.exp(-2j * np.pi * cfg.delta * (np.outer(egrid[:, 0], xp[a] - posg[a, 0]) + np.outer(egrid[:, 1], yp[a] - posg[a, 1])))
        kc3[a] = np.linalg.lstsq(B * wts[:, None], (B @ kc[a]) / Jtrue[i] * wts, rcond=None)[0]
    img_c = image(E.build_gridop(cfg, chan, kc_override=kc3), R[held], phi)
    # screen error at the source directions
    js = Jt(np.array([pos["CygA"], pos["CasA"]]))
    jf = beta @ basis(np.array([pos["CygA"], pos["CasA"]]))
    q = jf / js
    q = q * np.conj(q.mean(0))[None, :] / np.abs(q.mean(0))[None, :]
    print(f"  solved screens at Cyg A / Cas A: phase error rms {np.degrees(np.angle(q)).std(0).round(1).tolist()} deg "
          f"(truth screen rms there {np.degrees(np.angle(js / js.mean(0))).std(0).round(1).tolist()} deg)")
    for name, img in (("(a) direction-independent only", img_a), ("(b) kernels from kernel-StEFCal", img_b),
                      ("(c) kernels from the true screens (bound)", img_c)):
        sc = np.sum(img[vis] * ideal[vis]) / np.sum(img[vis] ** 2)
        err = (sc * img - ideal) * jy
        print(f"  {name:44s} error rms {err[vis].std():7.1f} Jy over the sky (held-out LST)", flush=True)
    # (d) image-domain prediction of the static sky: the actual image (DI weights, production
    # kernels) minus its prediction from the model sky, without and with the solved screens
    enu_m, S_m = model[held]
    pred_di = image(op0, vis_from(enu_m, S_m, g_di[:, None]), phi / g_di)
    pred_dd = image(op0, vis_from(enu_m, S_m, beta @ basis(enu_m)), phi / g_di)
    pred_tr = image(op0, vis_from(enu_m, S_m, Jt(enu_m)), phi / g_di)
    for name, pr in (("(d) predicted with DI gains only", pred_di), ("(d) predicted with kernel-StEFCal screens", pred_dd),
                     ("(d) predicted with the true screens", pred_tr)):
        res = (img_a - pr) * jy
        print(f"  {name:44s} residual rms {res[vis].std():7.1f} Jy (actual image minus prediction)", flush=True)
    np.save("kernelcal_beta.npy", beta)


if __name__ == "__main__":
    main()
