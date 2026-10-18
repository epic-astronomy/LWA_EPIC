"""Correlate one TBF capture into two independent halves (even / odd samples) for direction-dependent tests.

    python dd_cache.py FILE OUT.npz [NSAMP]
Saves R_even, R_odd [chan, pol, 256, 256] complex64 at 16 channels (28.5-46.5 MHz) and the capture time.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import tbf_cal as TC  # noqa: E402

path, out = sys.argv[1], sys.argv[2]
nsamp = int(sys.argv[3]) if len(sys.argv) > 3 else 24000
chans = np.arange(1140, 1900, 48)
T0 = time.time()
X, nvalid, t = TC.load_tbf(path, chans, nsamp)
x = X.transpose(0, 2, 1, 3)  # chan, pol, stand, t
Re = np.array([[xp[:, 0::2] @ xp[:, 0::2].conj().T / xp[:, 0::2].shape[1] for xp in xc] for xc in x]).astype(np.complex64)
Ro = np.array([[xp[:, 1::2] @ xp[:, 1::2].conj().T / xp[:, 1::2].shape[1] for xp in xc] for xc in x]).astype(np.complex64)
np.savez(out, R_even=Re, R_odd=Ro, chans=chans, t=t.isot, nsamp=X.shape[-1] // 2)
print(out, t.isot, f"{time.time() - T0:.0f}s")
