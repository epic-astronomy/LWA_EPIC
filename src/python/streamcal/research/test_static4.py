import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streamcal import sky as skymodel, epicop, forward, solvers as calib  # noqa: E402,F401
sys.modules.update({"skymodel": skymodel, "epicop": epicop, "forward": forward, "calib": calib})
import time, numpy as np
import forward as F, epicop as E, calib as C, simutil as U
rng = np.random.default_rng(2)
cfg = E.EpicConfig(); chan = 3168
op = E.build_gridop(cfg, chan); n = len(op.ant)
z = np.load('../static_75.npz'); Mt, Mm = z['Mt'], z['Mm']
eps = np.exp(rng.normal(0, 0.2, n) + 1j * rng.normal(0, np.radians(40), n))
Rt = eps[:, None] * (Mt + 0.15*np.real(np.diag(Mt)).mean()*np.eye(n)) * eps.conj()[None, :]
phi = F.geometric_phase(chan, op.ant)
T = int(5 * E.CHAN_BW); Rh = U.wishart(Rt, T, rng)
for umin_l in [1, 3]:
  for wname, mask in [('uniform', C.uv_mask(cfg.grid_size, umin_l/cfg.delta, 0)), ('ML', C.ml_weights(op, umin_l/cfg.delta))]:
    prob = C.GridProblem(op, (phi[op.a] * np.conj(phi[op.b])) * Mm[op.a, op.b], mask)
    for lab, R in [('exact', Rt), ('5s', Rh)]:
        D = E.scatter(op, (phi[:, None] * R * phi.conj()[None, :])[op.a, op.b]) + E.autos(op, np.real(np.diag(R)))
        t0=time.time(); d, _, info = C.gridcal(prob, D, n_outer=8, n_cg=25)
        print(f'umin {umin_l} {wname:7s} {lab:5s} err amp+ph {U.gain_error(d, eps)[0]:.3f} phase {U.gain_error(d, eps)[1]:.2f} deg  {time.time()-t0:.1f}s')
