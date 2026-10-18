"""Validate dd_analyze on simulated captures: null test and injected direction-dependent gains.

Writes dd_sim_<label>.npz in the dd_cache format (raw convention, cable model included) for:
  null   DI gains only (no direction dependence)
  inj    plus per-antenna gains toward Cyg A different from those toward Cas A:
         phase rms PH deg and amplitude rms AM, smooth in frequency (fixed across the band)
then the files can be run through dd_analyze.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from astropy.time import Time  # noqa: E402

from streamcal import forward as F, geometry as G, solvers as C  # noqa: E402
import simutil as U  # noqa: E402
import tbf_cal as TC  # noqa: E402

PH = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
AM = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
t = Time("2023-09-17T04:00:01")
chans = np.arange(1140, 1900, 48)
freq = chans * 25e3
allst = np.arange(256)
rall = G.eg.stand_xyz()
rng = np.random.default_rng(5)
T = 12000
gdi = np.exp(rng.normal(0, 0.15, (2, 256)) + 1j * rng.normal(0, 0.5, (2, 256)))
ddp = rng.normal(0, np.radians(PH), (2, 256))
dda = np.exp(rng.normal(0, AM, (2, 256)))
ex = F.random_extra_sources()
for label, inject in (("null", False), ("inj", True)):
    Re = np.zeros((len(chans), 2, 256, 256), np.complex64)
    Ro = np.zeros_like(Re)
    for p, pol in enumerate(("XX", "YY")):
        snap = F.build_snapshot(t, freq / G.CHAN_BW, pol=pol, nside=128, extra_sources=ex, fluct_rms=0.1)
        for ci, f in enumerate(freq):
            Md = snap.matrix(allst, ci, names=["diffuse", "extra"])
            Rt = np.outer(gdi[p], gdi[p].conj()) * Md
            for c in snap.comps:
                if c.name in ("diffuse", "extra"):
                    continue
                gk = gdi[p] * (dda[p] * np.exp(1j * ddp[p]) if (inject and c.name == "CygA") else 1.0)
                Rt += np.outer(gk, gk.conj()) * C.point_matrix(rall, f, c.enu[0], c.flux[0, ci])
            Rt += 0.15 * np.real(np.diag(Rt)).mean() * np.eye(256)
            kap = TC.cable_correction(f, p)
            Rraw = Rt / np.outer(kap, kap.conj())
            Re[ci, p] = U.wishart(Rraw, T, rng)
            Ro[ci, p] = U.wishart(Rraw, T, rng)
    out = f"dd_sim_{label}.npz"
    np.savez(out, R_even=Re, R_odd=Ro, chans=chans, t=t.isot, nsamp=T)
    print("wrote", out, flush=True)
