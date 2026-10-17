"""Can per-antenna direction-dependent corrections improve EPIC images? A faithful test.

Truth: every stand has a smooth random direction-dependent phase screen J_a(l, m)
(correlation length THETA deg), scaled so that the rms of arg J(CygA) / J(CasA) over
stands is 20 deg (measured on LWA-SV TBF and TBN data near 40 MHz). The screen applies
to the diffuse sky and to the A-team alike. Calibration is direction-independent
(StEFCal against the nominal sky), as in streamcal; the per-stand gains toward Cyg A and
Cas A are measured with 3 deg noise.

Treatments, each compared with the ideal image (no screens, production kernels), in Jy:
  (a) direction-independent only
  (b) kernels: per stand, c_a(l) = 1 + sum_k (1 / d_a^k - 1) exp(-|l - l_k|^2 / 2 w^2),
      fitted into the 5x5 kernel by least squares (epicop.build_gridop(kc_override=...))
  (c) image-domain subtraction of the two sources made with their measured gains,
      replaced by their ideal images
Reported: rms over the visible sky and within 12 deg of the sources.
DD_SCREEN_ON=sources|diffuse applies the screens to one component only (error budget).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
from astropy.time import Time  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

from streamcal import epicop as E, forward as F, solvers as C  # noqa: E402

cfg = E.EpicConfig()
chan = 1440
f = chan * E.CHAN_BW
t = Time("2021-04-04T11:35:00")
rng = np.random.default_rng(7)
op0 = E.build_gridop(cfg, chan)
ant = op0.ant
n = len(ant)
r = F.stand_enu()[ant]
phi = F.geometric_phase(chan, ant)
snap = F.build_snapshot(t, [chan], nside=64)
src = {c.name: (c.enu[0], float(c.flux[0, 0])) for c in snap.comps if c.name in ("CygA", "CasA")}
dif = [c for c in snap.comps if c.name == "diffuse"][0]
ng = 96
lg = np.linspace(-1, 1, ng)


def screens(theta_deg):
    s = np.radians(theta_deg)
    k = np.fft.fftfreq(ng, d=lg[1] - lg[0])
    KX, KY = np.meshgrid(k, k)
    filt = np.exp(-2 * (np.pi * s) ** 2 * (KX ** 2 + KY ** 2))
    fld = np.array([np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(ng, ng))) * filt)) for _ in range(n)])
    fld /= fld.std()
    return fld  # unit-variance phase fields [stand, m, l]


def at(fld, enu):
    """Field values at directions enu [k, 3] (l = east, m = north)."""
    pts = np.c_[np.clip(enu[:, 1], -1, 1), np.clip(enu[:, 0], -1, 1)]
    return np.array([RegularGridInterpolator((lg, lg), fa)(pts) for fa in fld])  # [stand, k]


def image(op, R, w):
    return E.image_from_vis(op, w[:, None] * R * np.conj(w)[None, :], with_autos=False)


def truth_R(phase):
    """Visibilities with per-stand screens: phase(enu) -> [stand, k] radians."""
    on = os.environ.get("DD_SCREEN_ON", "all")  # all | sources | diffuse
    A = np.exp(2j * np.pi * f / 299792458.0 * (r @ dif.enu.T))
    if on in ("all", "diffuse"):
        A = A * np.exp(1j * phase(dif.enu))
    R = (A * dif.flux[:, 0]) @ A.conj().T
    for nm, (e, S) in src.items():
        a = np.exp(2j * np.pi * f / 299792458.0 * (r @ e))
        if on in ("all", "sources"):
            a = a * np.exp(1j * phase(e[None])[:, 0])
        R += S * np.outer(a, a.conj())
    return R


def kernel_fit(kc, xp, yp, pos, corr_fn):
    """Fit each stand's 5x5 kernel to k_a(l) c_a(l) over the visible sky."""
    L, M = np.meshgrid(lg, lg)
    vis = L ** 2 + M ** 2 < 0.98
    lv, mv = L[vis], M[vis]
    w = np.sqrt(np.sqrt(1 - lv ** 2 - mv ** 2))
    out = kc.copy()
    for i, a in enumerate(ant):
        B = np.exp(-2j * np.pi * cfg.delta * (np.outer(lv, xp[a] - pos[a, 0]) + np.outer(mv, yp[a] - pos[a, 1])))
        target = (B @ kc[a]) * corr_fn(i, lv, mv)
        out[a] = np.linalg.lstsq(B * w[:, None], target * w, rcond=None)[0]
    return out


def main():
    kc, xp, yp, pos, used = E.kernels(cfg, chan)
    ideal_op = op0
    R0 = truth_R(lambda e: np.zeros((n, len(e))))
    ideal = image(ideal_op, R0, phi)
    P1 = C.point_matrix(r, f, src["CygA"][0], 1.0)
    jy = 1.0 / image(ideal_op, P1, phi).max()
    N = op0.N
    kk = np.fft.fftfreq(N) * N
    # pixel directions of the (unshifted) image: l = delta-grid frequency
    lpix = kk / (N * cfg.delta)
    LL, MM = np.meshgrid(lpix, lpix, indexing="ij")
    vis = LL ** 2 + MM ** 2 < 0.95
    near = np.zeros_like(vis)
    for e, _ in src.values():
        near |= np.hypot(LL - e[0], MM - e[1]) < np.sin(np.radians(12))
    print(f"34.5 MHz, {t.isot}: Cyg A {src['CygA'][1]:.0f} Jy, Cas A {src['CasA'][1]:.0f} Jy (apparent)")
    for theta in (10, 20, 40):
        fld = screens(theta)
        es = np.array([src["CygA"][0], src["CasA"][0]])
        v = at(fld, es)
        scale = np.radians(20) / np.std(v[:, 0] - v[:, 1])
        phase = lambda e: scale * at(fld, e)  # noqa: E731
        R = truth_R(phase)
        M_nom = R0
        W = np.ones((n, n)) - np.eye(n)
        g_di, _, _ = C.stefcal(R, M_nom, W, niter=200)
        g_di = C.reference_phase(g_di)
        w = phi / g_di
        img_a = image(op0, R, w)
        # directional gains toward each source relative to DI, measured with 3 deg noise
        d = {}
        for k, (nm, (e, S)) in enumerate(src.items()):
            d[nm] = np.exp(1j * phase(e[None])[:, 0]) / g_di * np.exp(1j * rng.normal(0, np.radians(3), n))
        res = {"(a) direction-independent only": img_a}
        for bump in (10, 20):
            sb = np.sin(np.radians(bump))

            def corr_fn(i, lv, mv, sb=sb):
                c = np.ones_like(lv, complex)
                for nm, (e, S) in src.items():
                    c += (1 / d[nm][i] - 1) * np.exp(-((lv - e[0]) ** 2 + (mv - e[1]) ** 2) / (2 * sb ** 2))
                return c

            op_b = E.build_gridop(cfg, chan, kc_override=kernel_fit(kc, xp, yp, pos, corr_fn))
            res[f"(b) 5x5 kernels, {bump} deg bumps"] = image(op_b, R, w)
        img_c = img_a.copy()
        for nm, (e, S) in src.items():
            a_k = np.exp(2j * np.pi * f / 299792458.0 * (r @ e))
            pred = S * np.outer(g_di * d[nm] * a_k, np.conj(g_di * d[nm] * a_k))
            img_c += -image(op0, pred, w) + image(op0, S * np.outer(a_k, a_k.conj()), phi)
        res["(c) image-domain subtraction"] = img_c
        print(f"  screen scale {theta} deg:")
        for k, img in res.items():
            sc = np.sum(img[vis] * ideal[vis]) / np.sum(img[vis] ** 2)
            err = (sc * img - ideal) * jy
            print(f"    {k:36s} error rms {err[vis].std():6.1f} Jy (sky), {err[near].std():6.1f} Jy (within 12 deg of the sources)")


if __name__ == "__main__":
    main()
