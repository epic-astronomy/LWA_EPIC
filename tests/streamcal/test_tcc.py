"""Check streamcal.tcc against numpy on random 8-bit data (run on a GPU node)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "python"))
import numpy as np  # noqa: E402

from streamcal import tcc  # noqa: E402

rng = np.random.default_rng(0)
x = (rng.integers(-20, 21, (2, 256, 2, 4096)) + 1j * rng.integers(-20, 21, (2, 256, 2, 4096))).astype(np.complex64)
R, nt, msg = tcc.correlate_i8(x)
print(msg)
ok = True
for c in range(2):
    for pY in range(2):
        for pX in range(2):
            ref = x[c, :, pY, :nt] @ x[c, :, pX, :nt].conj().T
            e = np.abs(R[c, pY, pX] - ref).max()
            ok &= e == 0
            print(f"chan {c} pol {pY}{pX}: max |TCC - numpy| = {e}")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
