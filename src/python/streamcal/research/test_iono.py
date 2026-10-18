"""Gain bias from ionospheric/solar changes of the bright sources: fixed model, free fluxes, free fluxes + positions."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streamcal import sky as skymodel, epicop, forward, solvers as calib  # noqa: E402,F401
sys.modules.update({"skymodel": skymodel, "epicop": epicop, "forward": forward, "calib": calib})
import numpy as np
from astropy.time import Time
import forward as F, epicop as E, calib as C, simutil as U
cfg = E.EpicConfig()
rng = np.random.default_rng(5)
for chan, tstr in [(3168, '2026-10-16T06:00:00'), (1440, '2026-10-16T19:00:00')]:
    op = E.build_gridop(cfg, chan); n = len(op.ant); t = Time(tstr)
    f = chan * E.CHAN_BW
    W = C.baseline_mask(op, n, 1 / cfg.delta)
    eps = np.exp(rng.normal(0, 0.2, n) + 1j * rng.normal(0, np.radians(40), n))
    mo = F.build_snapshot(t, [chan], nside=64)
    names = [c.name for c in mo.comps if c.name != 'diffuse']
    Mpt = {k: mo.matrix(op.ant, 0, names=[k]) for k in names}
    Mdiff = mo.matrix(op.ant, 0, names=['diffuse'])
    cases = {
        'no change': dict(),
        'Cyg A scintillates x0.6': dict(flux_scale={'CygA': 0.6}),
        'Cyg A x1.5, Cas A x0.7': dict(flux_scale={'CygA': 1.5, 'CasA': 0.7}),
        'refraction 0.3 deg x (74/f)^2': dict(offsets_deg={'CygA': (0.3 * (74e6 / f) ** 2, 0.0), 'CasA': (0.0, -0.3 * (74e6 / f) ** 2)}),
        'refraction + Cyg A x0.6': dict(offsets_deg={'CygA': (0.3 * (74e6 / f) ** 2, 0.0), 'CasA': (0.0, -0.3 * (74e6 / f) ** 2)}, flux_scale={'CygA': 0.6}),
    }
    if 'Sun' in names:
        cases['solar burst, Sun x30'] = dict(sun_flux_factor=30)
        cases['burst x30 + Cyg A x0.6'] = dict(sun_flux_factor=30, flux_scale={'CygA': 0.6})
    print(f'--- {f/1e6:.1f} MHz, {tstr[11:16]} UTC, sources up: {names}')
    for lab, kw in cases.items():
        tr = F.build_snapshot(t, [chan], nside=64, **kw)
        Mt = tr.matrix(op.ant, 0)
        Rt = eps[:, None] * Mt * eps.conj()[None, :]
        g1, _, _ = C.stefcal(Rt, mo.matrix(op.ant, 0), W, niter=300)
        srcs = [(c.enu[0], c.flux[0, 0]) for c in mo.comps if c.name != 'diffuse']
        r = F.stand_enu()[op.ant]
        g2, s2, _, _ = C.stefcal_sky(Rt, Mdiff, srcs, W, r, f, solve_pos=False, n_outer=60)
        g3, s3, o3, _ = C.stefcal_sky(Rt, Mdiff, srcs, W, r, f, solve_pos=True, n_outer=60)
        e1, e2, e3 = [U.gain_error_tiptilt(g, eps, r, f) for g in (g1, g2, g3)]
        print(f'  {lab:32s} fixed: {e1[1]:6.2f} deg {e1[0]:.3f} sh {e1[2]:.2f} | free flux: {e2[1]:6.2f} deg {e2[0]:.3f} sh {e2[2]:.2f} | '
              f'flux+pos: {e3[1]:5.2f} deg {e3[0]:.3f} sh {e3[2]:.2f}  flux {np.round(s3, 2)}')
