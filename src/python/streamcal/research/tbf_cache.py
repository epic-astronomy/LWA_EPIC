import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streamcal import sky as skymodel, epicop, forward, solvers as calib  # noqa: E402,F401
sys.modules.update({"skymodel": skymodel, "epicop": epicop, "forward": forward, "calib": calib})
import sys, numpy as np, time
import tbf_cal as TC
path, out = sys.argv[1], sys.argv[2]
nsamp = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
chans = np.arange(1140, 1900, 48)
T0 = time.time()
X, nvalid, t = TC.load_tbf(path, chans, nsamp)
R = np.einsum('csit,csjt->cisj', X.transpose(0,2,1,3).reshape(len(chans),2,256,-1)[:, :, None].squeeze(2), X.transpose(0,2,1,3).reshape(len(chans),2,256,-1).conj(), optimize=True) if False else None
x = X.transpose(0, 2, 1, 3)  # chan, pol, stand, t
R = np.array([[xp @ xp.conj().T / xp.shape[1] for xp in xc] for xc in x])  # chan, pol, 256, 256
np.savez(out, R=R, chans=chans, t=t.isot, nsamp=X.shape[-1])
print(out, t.isot, R.shape, f'{time.time()-T0:.0f}s')
