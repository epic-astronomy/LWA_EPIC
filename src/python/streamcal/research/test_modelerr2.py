import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streamcal import sky as skymodel, epicop, forward, solvers as calib  # noqa: E402,F401
sys.modules.update({"skymodel": skymodel, "epicop": epicop, "forward": forward, "calib": calib})
import time, numpy as np
from astropy.time import Time
import forward as F, epicop as E, calib as C, simutil as U
rng = np.random.default_rng(2)
cfg = E.EpicConfig()
t = Time('2026-10-16T06:00:00')
ex = F.random_extra_sources()
fs = {'CygA':1.08,'CasA':0.93,'TauA':1.1}
for chan in [3168, 1440]:   # 75.8 and 34.5 MHz
    op = E.build_gridop(cfg, chan); n = len(op.ant)
    T0=time.time()
    true128 = F.build_snapshot(t, [chan], nside=128, extra_sources=ex, flux_scale=fs)
    truefl = F.build_snapshot(t, [chan], nside=128, extra_sources=ex, flux_scale=fs, fluct_rms=0.1)
    Mt = true128.matrix(op.ant, 0); Mtf = truefl.matrix(op.ant, 0)
    print('chan', chan, 'truth build', round(time.time()-T0,1), 's')
    models = {'nside64': F.build_snapshot(t, [chan], nside=64).matrix(op.ant, 0),
              'nside32': F.build_snapshot(t, [chan], nside=32).matrix(op.ant, 0)}
    eps = np.exp(rng.normal(0, 0.2, n) + 1j * rng.normal(0, np.radians(40), n))
    for tl, M in [('LFSM truth', Mt), ('LFSM+10% fluct truth', Mtf)]:
        Rt = eps[:, None] * (M + 0.15*np.real(np.diag(M)).mean()*np.eye(n)) * eps.conj()[None, :]
        for umin_lam in [1, 3, 6]:
            W = C.baseline_mask(op, n, umin_lam / cfg.delta)
            for k, Mm in models.items():
                g, s, it = C.stefcal(Rt, Mm, W, niter=300)
                e = U.gain_error(g, eps); print(f'  {tl:22s} umin {umin_lam}  model {k}: amp+phase rms {e[0]:.3f}  phase {e[1]:.2f} deg  it {it}')
