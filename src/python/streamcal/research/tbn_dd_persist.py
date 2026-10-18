"""Persistence of the per-antenna direction-dependent phase over 20 minutes (TBN, one channel).

Input: dd_analyze rows for 10 s chunks. For each pair of chunks (i, j) and polarisation,
the noise-free correlation of the per-stand phase of g(CygA)/g(CasA),
    rho_ij = cov(even_i, odd_j) / sqrt(cov(even_i, odd_i) cov(even_j, odd_j)),
uses only cross-half products. Reported against the time separation, with the
per-chunk signal and noise and the X/Y correlation.
    python tbn_dd_persist.py dd_tbn.npy
"""
import sys

import numpy as np
from astropy.time import Time

rows = []
for fn in sys.argv[1:]:
    rows += list(np.load(fn, allow_pickle=True))
cov = lambda a, b: np.nanmean((a - np.nanmean(a)) * (b - np.nanmean(b)))
for pol in ("XX", "YY"):
    rp = sorted([r_ for r_ in rows if r_["pol"] == pol], key=lambda r_: r_["t"])
    if not rp:
        continue
    t = np.array([Time(r_["t"]).unix for r_ in rp])
    sg = [1.0 if tuple(r_["pair"]) == ("CygA", "CasA") else -1.0 for r_ in rp]
    E = np.array([np.where(r_["mask"][0], r_["dd_even"][0].imag, np.nan) * s for r_, s in zip(rp, sg)])
    O = np.array([np.where(r_["mask"][0], r_["dd_odd"][0].imag, np.nan) * s for r_, s in zip(rp, sg)])
    sig = np.array([cov(E[i], O[i]) for i in range(len(rp))])
    noi = np.array([np.nanmean((E[i] - O[i]) ** 2) / 2 for i in range(len(rp))])
    print(f"{pol}: {len(rp)} chunks; per-chunk DD phase signal {np.degrees(np.sqrt(np.median(np.maximum(sig, 0)))):.1f} deg rms, "
          f"noise {np.degrees(np.sqrt(np.median(noi))):.1f} deg")
    bins = [(0, 15), (15, 60), (60, 200), (200, 500), (500, 1300)]
    for lo, hi in bins:
        rh = []
        for i in range(len(rp)):
            for j in range(len(rp)):
                if i == j or not (lo < abs(t[i] - t[j]) <= hi) or sig[i] <= 0 or sig[j] <= 0:
                    continue
                rh.append(cov(E[i], O[j]) / np.sqrt(sig[i] * sig[j]))
        if rh:
            print(f"   separation {lo:4d}-{hi:4d} s: correlation {np.median(rh):+.2f} ({len(rh)} pairs)")
xy = {}
for r_ in rows:
    s = 1.0 if tuple(r_["pair"]) == ("CygA", "CasA") else -1.0
    xy.setdefault(r_["t"], {})[r_["pol"]] = (np.where(r_["mask"][0], r_["dd_even"][0].imag, np.nan) * s,
                                             np.where(r_["mask"][0], r_["dd_odd"][0].imag, np.nan) * s)
c = [cov(d["XX"][0], d["YY"][1]) / np.sqrt(cov(*d["XX"]) * cov(*d["YY"]))
     for d in xy.values() if "XX" in d and "YY" in d and cov(*d["XX"]) > 0 and cov(*d["YY"]) > 0]
if c:
    print(f"X/Y correlation of the DD phase (noise-free): median {np.median(c):+.2f} over {len(c)} chunks")
