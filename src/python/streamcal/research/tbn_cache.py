"""Correlate a raw TBN capture (one 100 kHz channel, 512 inputs) in chunks, each split into even/odd halves.

    python tbn_cache.py FILE OUTDIR START_S NCHUNK [CHUNK_S]
Writes OUTDIR/chunk_<start>.npz with R_even, R_odd [1, 2, 256, 256], freq, t, nsamp (the dd_analyze format).
Input order from lsl's TBNFile.read is stand-major (2 * (stand - 1) + pol), as lwasv.antennas.
"""
import os
import sys
import time

import numpy as np
from lsl.reader.ldp import LWADataFile

fn, outdir, start, nchunk = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
chunk = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
os.makedirs(outdir, exist_ok=True)
f = LWADataFile(fn)
freq = f.get_info("freq1")
f.offset(start)
for k in range(nchunk):
    T0 = time.time()
    out = os.path.join(outdir, f"chunk_{start + k * chunk:07.1f}.npz")
    if os.path.exists(out):  # resume: skip chunks already written
        f.offset(chunk)
        continue
    tint, tstart, d = f.read(chunk)
    x = d.reshape(256, 2, -1).transpose(1, 0, 2)  # pol, stand, t
    Re = np.array([xp[:, 0::2] @ xp[:, 0::2].conj().T / xp[:, 0::2].shape[1] for xp in x])[None].astype(np.complex64)
    Ro = np.array([xp[:, 1::2] @ xp[:, 1::2].conj().T / xp[:, 1::2].shape[1] for xp in x])[None].astype(np.complex64)
    np.savez(out, R_even=Re, R_odd=Ro, freq=np.array([freq]), chans=np.array([0]), t=str(tstart.datetime.isoformat()),
             nsamp=d.shape[1] // 2)
    print(out, tstart, d.shape, f"{time.time() - T0:.0f}s", flush=True)
