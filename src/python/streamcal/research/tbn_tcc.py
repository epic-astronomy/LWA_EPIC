"""tbn_cache.py on the GPU: correlate TBN chunks (even / odd halves) with the Tensor-Core Correlator.

    python tbn_tcc.py FILE OUTDIR START_S NCHUNK [CHUNK_S]
Same output as tbn_cache.py (R_even, R_odd [1, 2, 256, 256], freq, t, nsamp); skips chunks already written.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
from lsl.reader.ldp import LWADataFile  # noqa: E402

from streamcal import tcc  # noqa: E402

fn, outdir, start, nchunk = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
chunk = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
os.makedirs(outdir, exist_ok=True)
f = LWADataFile(fn)
freq = f.get_info("freq1")
f.offset(start)
for k in range(nchunk):
    out = os.path.join(outdir, f"chunk_{start + k * chunk:07.1f}.npz")
    if os.path.exists(out):
        f.offset(chunk)
        continue
    T0 = time.time()
    tint, tstart, d = f.read(chunk)
    T1 = time.time()
    x = d.reshape(256, 2, -1)[None]  # [chan 1, stand, pol, t]
    halves = []
    for h in (0, 1):
        Rh, nt, msg = tcc.correlate_i8(np.ascontiguousarray(x[..., h::2]))
        halves.append((Rh[:, [0, 1], [0, 1]] / nt).astype(np.complex64))  # XX, YY: [1, 2, 256, 256]
    np.savez(out, R_even=halves[0], R_odd=halves[1], freq=np.array([freq]), chans=np.array([0]),
             t=str(tstart.datetime.isoformat()), nsamp=nt)
    print(out, tstart, f"read {T1 - T0:.0f}s, correlate {time.time() - T1:.1f}s", flush=True)
