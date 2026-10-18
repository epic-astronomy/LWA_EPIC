import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streamcal import sky as skymodel, epicop, forward, solvers as calib  # noqa: E402,F401
sys.modules.update({"skymodel": skymodel, "epicop": epicop, "forward": forward, "calib": calib})
import numpy as np, time
from astropy.time import Time
import forward as F, epicop as E, calib as C, simutil as U
cfg = E.EpicConfig()
for chan, tstr in [(1440, '2026-10-16T06:00:00'), (1440, '2026-10-16T19:00:00')]:
    op = E.build_gridop(cfg, chan); n = len(op.ant)
    t = Time(tstr)
    Mt = F.build_snapshot(t, [chan], nside=128, extra_sources=F.random_extra_sources(), fluct_rms=0.1, sun_flux_factor=3).matrix(op.ant, 0)
    mo = F.build_snapshot(t, [chan], nside=64); Mm = mo.matrix(op.ant, 0); Ms = mo.matrix(op.ant, 0, names=['Sun'])
    rng = np.random.default_rng(3)
    eps = np.exp(rng.normal(0, 0.2, n) + 1j * rng.normal(0, np.radians(40), n))
    Rt = eps[:, None] * (Mt + 0.15*np.real(np.diag(Mt)).mean()*np.eye(n)) * eps.conj()[None, :]
    phi = F.geometric_phase(chan, op.ant)
    D = E.scatter(op, (phi[:, None] * Rt * phi.conj()[None, :])[op.a, op.b])
    lv = phi[op.a] * np.conj(phi[op.b])
    mask = C.ml_weights(op, 1/cfg.delta)
    W = C.baseline_mask(op, n, 1/cfg.delta)
    sun_up = np.abs(Ms).max() > 0
    for lab, prob in [('fixed model', C.GridProblem(op, lv*Mm[op.a, op.b], mask)),
                      ('Sun free', C.GridProblem(op, lv*(Mm-Ms)[op.a, op.b], mask, free=[lv*Ms[op.a, op.b]]) if sun_up else None)]:
        if prob is None: continue
        for no, ncg in [(2, 15), (10, 40)]:
            d, s, info = C.gridcal(prob, D, n_outer=no, n_cg=ncg)
            print(f'{tstr[11:16]} {chan*E.CHAN_BW/1e6:.1f} MHz grid {lab:11s} outer {no:2d}: err {U.gain_error(d, eps)[1]:6.2f} deg  sun scale {s}')
    g,_,_ = C.stefcal(Rt, Mm - Ms, W, free=[Ms]) if sun_up else C.stefcal(Rt, Mm, W)
    print(f'{tstr[11:16]} stefcal exact R: err {U.gain_error(g, eps)[1]:.2f} deg')
