"""What is the measured per-antenna direction-dependent phase? Discriminating tests.

For each capture (dd_analyze rows), with phi[c, a] = arg(g_a(CygA)/g_a(CasA)):
  X/Y      correlation of the band-averaged phases of the X and Y dipoles of each stand
           (shared by the stand: ionosphere, sky-model error, geometry; different:
           element patterns and electronics)
  1/f vs f per stand, fit phi(f) = alpha / f and phi(f) = beta f (each with a per-channel
           constant already removed by the tip/tilt normalisation) and compare residuals:
           the ionosphere scales as 1/f, geometric path differences as f
  t        correlation of the band-averaged phases between consecutive captures (30 min):
           static instrumental effects persist, the ionosphere does not
    python dd_diagnose.py dd_results_*.npy
"""
import sys

import numpy as np

rows = []
for fn in sys.argv[1:]:
    rows += list(np.load(fn, allow_pickle=True))
by = {}
for r_ in rows:
    sgn = 1.0 if tuple(r_["pair"]) == ("CygA", "CasA") else -1.0  # orient every ratio as CygA / CasA
    by.setdefault(r_["t"], {})[r_["pol"]] = (np.imag(r_["dd"]) * sgn, r_["freq"])
times = sorted(by)
print("capture   X/Y corr (band-avg)   1/f better than f (fraction of stands)   corr with previous capture X / Y")
prev = None
for t in times:
    d = by[t]
    line = f"{t[11:16]}   "
    if "XX" in d and "YY" in d:
        bx, by_ = np.nanmean(d["XX"][0], 0), np.nanmean(d["YY"][0], 0)
        ok = np.isfinite(bx) & np.isfinite(by_)
        line += f"{np.corrcoef(bx[ok], by_[ok])[0, 1]:+.2f}               "
    else:
        line += "  -                 "
    fr = []
    for pol in d:
        ph, f = d[pol]
        if len(f) < 6:
            continue
        x1, x2 = 1 / f, f
        for a in range(ph.shape[1]):
            y = ph[:, a]
            ok = np.isfinite(y)
            if ok.sum() < 6:
                continue
            res = []
            for x in (x1, x2):
                X = np.c_[x[ok] - x[ok].mean()]
                c = np.linalg.lstsq(X, y[ok] - y[ok].mean(), rcond=None)[0]
                res.append(np.sum((y[ok] - y[ok].mean() - X @ c) ** 2))
            fr.append(res[0] < res[1])
    line += f"{np.mean(fr):.2f}                                  " if fr else "-                                     "
    if prev is not None:
        cs = []
        for pol in ("XX", "YY"):
            if pol in d and pol in prev:
                a, b = np.nanmean(d[pol][0], 0), np.nanmean(prev[pol][0], 0)
                ok = np.isfinite(a) & np.isfinite(b)
                cs.append(f"{np.corrcoef(a[ok], b[ok])[0, 1]:+.2f}")
        line += " / ".join(cs)
    print(line)
    prev = d
