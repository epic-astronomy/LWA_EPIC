"""Frequency scaling of the per-antenna direction-dependent phase (noise-unbiased).

For each capture and polarisation the band is split into its lowest and highest
quarters; per stand the mean phase in each quarter is formed for the even and the odd
halves of the data. The slope of phi_low against phi_high over stands,
    slope = cov(low_even, high_odd) / cov(high_even, high_odd),
uses only cross-half covariances, so independent noise does not bias it.
The ionosphere predicts slope = f_high / f_low (phase ~ 1/f); a geometric path
difference predicts f_low / f_high; a frequency-independent phase predicts 1.
    python dd_freqscale.py dd_results_*.npy
"""
import sys

import numpy as np

rows = []
for fn in sys.argv[1:]:
    rows += [r_ for r_ in np.load(fn, allow_pickle=True) if "dd_even" in r_]
for r_ in sorted(rows, key=lambda x: (x["t"], x["pol"])):
    f = r_["freq"]
    if len(f) < 8:
        continue
    q = len(f) // 4
    lo_i, hi_i = np.arange(q), np.arange(len(f) - q, len(f))
    m = r_["mask"]
    pe = np.where(m, r_["dd_even"].imag, np.nan)
    po = np.where(m, r_["dd_odd"].imag, np.nan)
    le, lo_ = np.nanmean(pe[lo_i], 0), np.nanmean(po[lo_i], 0)
    he, ho = np.nanmean(pe[hi_i], 0), np.nanmean(po[hi_i], 0)
    ok = np.isfinite(le) & np.isfinite(lo_) & np.isfinite(he) & np.isfinite(ho)
    cov = lambda a, b: np.mean((a - a.mean()) * (b - b.mean()))
    slope = 0.5 * (cov(le[ok], ho[ok]) + cov(lo_[ok], he[ok])) / cov(he[ok], ho[ok])
    corr = 0.5 * (cov(le[ok], ho[ok]) + cov(lo_[ok], he[ok])) / np.sqrt(cov(le[ok], lo_[ok]) * cov(he[ok], ho[ok]))
    fl, fh = f[lo_i].mean(), f[hi_i].mean()
    print(f"{r_['t'][11:16]} {r_['pol']}: low {fl / 1e6:.1f} MHz vs high {fh / 1e6:.1f} MHz: slope {slope:+.2f} "
          f"(ionosphere {fh / fl:.2f}, geometry {fl / fh:.2f}, flat 1.00); noise-free correlation {corr:+.2f}")
