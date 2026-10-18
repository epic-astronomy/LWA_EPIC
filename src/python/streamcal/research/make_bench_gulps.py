"""Step 2 of the benchmark dataset: 4+4-bit F-engine gulps from the covariance factors.

Writes NGULPS gulps of [chan][seq 1000][stand 256][pol 2] CNib bytes (re in the high nibble,
im in the low nibble, two's complement), the layout epic++ images (CHAN_MAJOR). Only numpy
is needed, so it runs on the GPU node into /dev/shm.

    python make_bench_gulps.py MODELDIR /dev/shm/streamcal_bench/gulps.bin --ngulps 96
"""
import argparse
import os
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--ngulps", type=int, default=96)
ap.add_argument("--nseq", type=int, default=1000)
ap.add_argument("--rms", type=float, default=2.0, help="rms per real component at the median input power")
ap.add_argument("--seed", type=int, default=3)
a = ap.parse_args()
L = np.load(os.path.join(a.model, "chol.npy"))  # [chan, pol, 256, 256]
nchan = L.shape[0]
pw = np.einsum("cpij,cpij->cpi", L, L.conj()).real  # input powers
scale = a.rms / np.sqrt(np.median(pw[pw > 0]) / 2)
L = (L * scale).astype(np.complex64)
rng = np.random.default_rng(a.seed)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
T0 = time.time()
with open(a.out, "wb") as fh:
    for k in range(a.ngulps):
        g = np.empty((nchan, a.nseq, 256, 2), np.uint8)
        for c in range(nchan):
            for p in range(2):
                z = ((rng.standard_normal((256, a.nseq), np.float32) + 1j * rng.standard_normal((256, a.nseq), np.float32))
                     / np.sqrt(2)).astype(np.complex64)
                x = L[c, p] @ z  # [256, nseq]
                re = np.clip(np.rint(x.real), -7, 7).astype(np.int8)
                im = np.clip(np.rint(x.imag), -7, 7).astype(np.int8)
                g[c, :, :, p] = (((re.astype(np.uint8) & 0xF) << 4) | (im.astype(np.uint8) & 0xF)).T
        g.tofile(fh)
        if k % 8 == 0:
            print(f"gulp {k}/{a.ngulps} ({time.time() - T0:.0f}s)", flush=True)
print("wrote", a.out, os.path.getsize(a.out) / 1e9, "GB")
