import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streamcal import sky as skymodel, epicop, forward, solvers as calib  # noqa: E402,F401
sys.modules.update({"skymodel": skymodel, "epicop": epicop, "forward": forward, "calib": calib})
import numpy as np
import forward as F, epicop as E, calib as C, simutil as U
cfg = E.EpicConfig(); chan = 3168
op = E.build_gridop(cfg, chan); n = len(op.ant)
Mt = np.load('../static_75.npz')['Mt']
phi = F.geometric_phase(chan, op.ant)
umin = 1/cfg.delta
W = C.baseline_mask(op, n, umin); mask = C.ml_weights(op, umin)
prob = C.GridProblem(op, (phi[op.a] * np.conj(phi[op.b])) * Mt[op.a, op.b], mask)
res = {}
for trial in range(3):
    rng = np.random.default_rng(10+trial)
    eps = np.exp(rng.normal(0, 0.2, n) + 1j * rng.normal(0, np.radians(40), n))
    Rt = eps[:, None] * (Mt + 0.15*np.real(np.diag(Mt)).mean()*np.eye(n)) * eps.conj()[None, :]
    for tint in [1, 5, 20]:
        T = int(tint * E.CHAN_BW)
        Rh = U.wishart(Rt, T, rng)
        D = E.scatter(op, (phi[:, None] * Rh * phi.conj()[None, :])[op.a, op.b])
        d,_,_ = C.gridcal(prob, D, n_outer=8, n_cg=25)
        res.setdefault(('images 100%', tint), []).append(U.gain_error(d, eps))
        g,_,_ = C.stefcal(Rh, Mt, W); res.setdefault(('vis 100%', tint), []).append(U.gain_error(g, eps))
        for duty in [0.01, 0.03, 0.1]:
            g,_,_ = C.stefcal(U.wishart(Rt, int(duty*T), rng), Mt, W)
            res.setdefault((f'tap {duty:.0%}', tint), []).append(U.gain_error(g, eps))
for k, v in sorted(res.items(), key=lambda x: (x[0][1], x[0][0])):
    v = np.mean(v, 0); print(f'{k[1]:3d} s  {k[0]:12s} amp+ph {v[0]:.4f}  phase {v[1]:.2f} deg')
