"""Step 1 of the benchmark dataset: true covariance factors with wrong gains.

For every channel of one band and both polarisations, builds the true sky covariance of all
256 stands (LFSM nside 128 with 10% small-scale structure, 25 faint sources absent from the
streamcal model, A-team, Sun), applies true residual gains eps and the inverse cable model in
the raw F-engine convention (x = eps / kappa * s, so that epic++'s nominal phases leave eps),
adds receiver noise, and stores the Cholesky factors. make_bench_gulps.py turns them into
4+4-bit gulps.

    python make_bench_model.py OUTDIR --chan0 3104 --time 2026-10-17T06:00:00
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
from astropy.time import Time  # noqa: E402

from streamcal import daemon as D, forward as F, geometry as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--chan0", type=int, default=3104)
ap.add_argument("--nchan", type=int, default=128)
ap.add_argument("--time", default="2026-10-17T06:00:00")
ap.add_argument("--seed", type=int, default=21)
ap.add_argument("--perfect", action="store_true", help="no residual gain errors (eps = 1, no dead stands)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(a.seed)
t = Time(a.time)
chans = a.chan0 + np.arange(a.nchan)
freq = chans * G.CHAN_BW
allst = np.arange(256)
nu_c = freq[a.nchan // 2]
# true residual gains [pol, stand, chan]
amp = np.exp(rng.normal(0, 0.2, (2, 256)))
ph0 = rng.uniform(-1, 1, (2, 256)) * np.radians(70)
tau = rng.normal(0, 3.0, (2, 256))  # ns
eps = amp[:, :, None] * np.exp(1j * (ph0[:, :, None] + 2 * np.pi * (freq - nu_c)[None, None, :] * 1e-9 * tau[:, :, None]))
used = np.flatnonzero(G.eg.used_stands())
dead = rng.choice(used, 3, replace=False)
eps[:, dead, :] *= 0.02
if a.perfect:
    eps = np.ones_like(eps)
L = np.zeros((a.nchan, 2, 256, 256), np.complex64)
ex = F.random_extra_sources()
for p, pol in enumerate(("XX", "YY")):
    snap = F.build_snapshot(t, chans, pol=pol, nside=128, extra_sources=ex, fluct_rms=0.1)
    for ci in range(a.nchan):
        M = snap.matrix(allst, ci)
        M = M + 0.15 * np.real(np.diag(M)).mean() * np.eye(256)
        kap = D.cable_correction(freq[ci], p)
        e = eps[p, :, ci] / kap
        Rt = e[:, None] * M * np.conj(e)[None, :]
        L[ci, p] = np.linalg.cholesky(Rt + 1e-6 * np.real(np.trace(Rt)) / 256 * np.eye(256))
        if ci % 32 == 0:
            print(pol, ci, flush=True)
np.save(os.path.join(a.out, "chol.npy"), L)
np.savez(os.path.join(a.out, "truth.npz"), eps=eps, chans=chans, dead=dead, time=a.time,
         seq0=int(t.unix / (8192 / 196e6)) // 1000 * 1000)
print("wrote", a.out)
