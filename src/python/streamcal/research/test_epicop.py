import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streamcal import sky as skymodel, epicop, forward, solvers as calib  # noqa: E402,F401
sys.modules.update({"skymodel": skymodel, "epicop": epicop, "forward": forward, "calib": calib})
import time, numpy as np, epicop as E
cfg = E.EpicConfig()
chan = 3168  # 75.8 MHz
t=time.time(); op = E.build_gridop(cfg, chan); print('build', time.time()-t, 's  nant', len(op.ant), 'nbl', len(op.a))
rng = np.random.default_rng(1)
n = len(op.ant); T = 40
X = (rng.normal(size=(n, T)) + 1j*rng.normal(size=(n, T)))
w = np.exp(1j*rng.uniform(0, 2*np.pi, n)) * rng.uniform(0.5, 1.5, n)
wX = w[:, None]*X
R = wX @ wX.conj().T
t=time.time(); img1 = E.image_from_vis(op, R); print('vis->img', time.time()-t)
img2 = E.moff_image(op, cfg, chan, w, X)
print('rel err', np.abs(img1-img2).max()/np.abs(img2).max())
# adjoint test
v = rng.normal(size=len(op.a))+1j*rng.normal(size=len(op.a)); D = rng.normal(size=(128,128))+1j*rng.normal(size=(128,128))
print('adjoint', np.vdot(E.scatter(op, v), D), np.vdot(v, E.gather(op, D)))
t=time.time(); [E.scatter(op, v) for _ in range(5)]; print('scatter ms', (time.time()-t)/5*1e3)
t=time.time(); [E.gather(op, D) for _ in range(5)]; print('gather ms', (time.time()-t)/5*1e3)
