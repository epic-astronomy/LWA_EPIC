"""Benchmark gulps with the covariance of a recorded GPU-tap table (real sky, gains and interference).

Each channel and polarisation gets voltages x = F z with F F^H = R (the table's covariance, from
its eigendecomposition; X and Y independent, since the table holds XX and YY only), quantised to
4+4 bits at the recorded level. Writes [gulp][chan][seq][stand][pol] CNib bytes for
epic++ --bench --bench_data, and MODELDIR/truth.npz (seq0, time) for bench_run.sh.

    python make_table_gulps.py TABLE OUT MODELDIR [--ngulps 48] [--nseq 1000]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
from astropy.time import Time  # noqa: E402

from streamcal.research.vistable_to_npz import SEQ_S, read  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("table")
ap.add_argument("out")
ap.add_argument("model")
ap.add_argument("--ngulps", type=int, default=48)
ap.add_argument("--nseq", type=int, default=1000)
ap.add_argument("--seed", type=int, default=5)
a = ap.parse_args()
t = read(a.table, 1)
R = t["R"]  # [chan, pol, 256, 256]
nchan = R.shape[0]
F = np.zeros_like(R)
for c in range(nchan):
    for p in range(2):
        w, v = np.linalg.eigh(R[c, p].astype(np.complex128))
        F[c, p] = (v * np.sqrt(np.clip(w, 0, None))[None, :]).astype(np.complex64)
pw = np.real(np.einsum("cpii->cpi", R))
print(f"{a.table}: {nchan} channels from {t['chans'][0]}, t {Time(t['t'], format='unix').isot}, median input "
      f"power {np.median(pw[pw > 0]):.1f} counts^2", flush=True)
seq0 = int(round(t["t"] / SEQ_S)) // a.nseq * a.nseq
os.makedirs(a.model, exist_ok=True)
np.savez(os.path.join(a.model, "truth.npz"), seq0=seq0, time=Time(seq0 * SEQ_S, format="unix").isot,
         chan0=int(t["chans"][0]), table=a.table)
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
                x = F[c, p] @ z
                re = np.clip(np.rint(x.real), -7, 7).astype(np.int8)
                im = np.clip(np.rint(x.imag), -7, 7).astype(np.int8)
                g[c, :, :, p] = (((re.astype(np.uint8) & 0xF) << 4) | (im.astype(np.uint8) & 0xF)).T
        g.tofile(fh)
        if k % 8 == 0:
            print(f"gulp {k}/{a.ngulps} ({time.time() - T0:.0f}s)", flush=True)
print("wrote", a.out, os.path.getsize(a.out) / 1e9, "GB; seq0", seq0)
