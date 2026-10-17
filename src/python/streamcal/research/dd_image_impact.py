"""What do uncorrected per-antenna direction-dependent gain errors cost in an EPIC image?

For a bright source k (Cyg A) the stands' gains toward k differ from the
direction-independent solution by exp(eps_a), eps complex Gaussian with rms sigma
(radians of phase and log-amplitude each). The error pattern is
    image(d d^H .* S_k P_k) - image(S_k P_k),
made with the exact epic++ image operator (epicop). It is static: it does not average
down with integration. It is compared with the thermal noise of one 83.6 ms image of
the full forward-modelled sky (Wishart draw, 2000 sequences, receiver noise 15%),
both in Jy (scale from a noiseless 1 Jy point source at the same position).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from astropy.time import Time  # noqa: E402

from streamcal import epicop as E, forward as F, solvers as C  # noqa: E402
import simutil as U  # noqa: E402

cfg = E.EpicConfig()
rng = np.random.default_rng(3)
t = Time("2026-10-16T06:00:00")
for chan in (1440, 3168):
    f = chan * E.CHAN_BW
    op = E.build_gridop(cfg, chan)
    n = len(op.ant)
    phi = F.geometric_phase(chan, op.ant)
    r = F.stand_enu()[op.ant]
    snap = F.build_snapshot(t, [chan], nside=64)
    cyg = [c for c in snap.comps if c.name == "CygA"][0]
    S = float(cyg.flux[0, 0])
    P1 = C.point_matrix(r, f, cyg.enu[0], 1.0)

    def img(R):
        return E.image_from_vis(op, phi[:, None] * R * np.conj(phi)[None, :], with_autos=False)

    ref1 = img(P1)
    jy = 1.0 / ref1.max()  # image units per Jy at the source
    N = op.N
    k = np.fft.fftfreq(N) * N
    vis = np.hypot(k[:, None], k[None, :]) < 0.45 * N  # rough visible-sky mask in the unshifted image
    # thermal noise of one image of the full sky
    Mt = snap.matrix(op.ant, 0)
    Rt = Mt + 0.15 * np.real(np.diag(Mt)).mean() * np.eye(n)
    noise = img(U.wishart(Rt, 2000, rng) - Rt) * jy
    sig_th = noise.std()
    print(f"{f / 1e6:.1f} MHz: Cyg A apparent {S:.0f} Jy; thermal noise per 83.6 ms image {sig_th:.0f} Jy (rms over the image)")
    for sig_deg in (1, 3, 10):
        s = np.radians(sig_deg)
        vals = []
        for trial in range(4):
            eps = rng.normal(0, s, n) + 1j * rng.normal(0, s, n)
            d = np.exp(eps)
            d /= np.mean(d)  # the DI solution absorbs the mean
            art = (img(S * np.outer(d, d.conj()) * P1) - img(S * P1)) * jy
            vals.append((art.std(), np.abs(art).max()))
        a_rms, a_max = np.mean(vals, 0)
        t_eq = (sig_th / a_rms) ** 2 * 0.0836
        print(f"   DD error {sig_deg:2d} deg rms: artefact rms {a_rms:7.1f} Jy, peak {a_max:7.0f} Jy "
              f"({a_rms / sig_th:.2f} x thermal per image; equals the thermal noise after {t_eq:,.1f} s of integration)")
