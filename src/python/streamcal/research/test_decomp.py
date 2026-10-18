import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streamcal import sky as skymodel, epicop, forward, solvers as calib  # noqa: E402,F401
sys.modules.update({"skymodel": skymodel, "epicop": epicop, "forward": forward, "calib": calib})
import time, numpy as np
from astropy.time import Time
import forward as F, epicop as E, calib as C, simutil as U
rng = np.random.default_rng(2)
cfg = E.EpicConfig(); chan = 3168
op = E.build_gridop(cfg, chan); n = len(op.ant)
t = Time('2026-10-16T06:00:00'); ex = F.random_extra_sources(); fs = {'CygA':1.08,'CasA':0.93,'TauA':1.1}
snapT = F.build_snapshot(t, [chan], nside=128, extra_sources=ex, flux_scale=fs, fluct_rms=0.1)
snap128 = F.build_snapshot(t, [chan], nside=128, extra_sources=ex, flux_scale=fs)
snap64 = F.build_snapshot(t, [chan], nside=64, extra_sources=ex, flux_scale=fs)
snapNom = F.build_snapshot(t, [chan], nside=128)
pts = ['CygA','CasA','TauA','VirA','Sun','extra']
cases = {
 'fluct only':  (snapT.matrix(op.ant,0), snap128.matrix(op.ant,0)),
 'pixel64 only': (snap128.matrix(op.ant,0), snap64.matrix(op.ant,0)),
 'points only': (snap128.matrix(op.ant,0), snapNom.matrix(op.ant,0)),
 'diffuse absent from model': (snap128.matrix(op.ant,0), snap128.matrix(op.ant,0,names=pts)),
}
eps = np.exp(rng.normal(0, 0.2, n) + 1j * rng.normal(0, np.radians(40), n))
phi = F.geometric_phase(chan, op.ant)
for umin_l in [1, 3]:
    W = C.baseline_mask(op, n, umin_l/cfg.delta); mask = C.ml_weights(op, umin_l/cfg.delta)
    for k, (Mt, Mm) in cases.items():
        Rt = eps[:, None] * Mt * eps.conj()[None, :]
        g,_,_ = C.stefcal(Rt, Mm, W, niter=300)
        prob = C.GridProblem(op, (phi[op.a] * np.conj(phi[op.b])) * Mm[op.a, op.b], mask)
        D = E.scatter(op, (phi[:, None] * Rt * phi.conj()[None, :])[op.a, op.b])
        d,_,_ = C.gridcal(prob, D, n_outer=8, n_cg=25)
        es, eg = U.gain_error(g, eps), U.gain_error(d, eps)
        print(f'umin {umin_l} {k:26s} stefcal {es[0]:.3f}/{es[1]:.2f}deg   gridcal {eg[0]:.3f}/{eg[1]:.2f}deg')
