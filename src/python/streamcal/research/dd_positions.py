"""Do the per-antenna direction-dependent phases come from stand position errors?

Input: dd_analyze result files (one row per capture and polarisation, with the
band of per-antenna log-ratios d = log(g_CygA / g_CasA), common phase, tip/tilt and
median amplitude removed). A position error Delta r_a (or an offset of the dipole's
electrical phase centre) gives
    arg d_a(f, t) = 2 pi f / c  Delta r_a . (s_CygA(t) - s_CasA(t))   (+ per-channel constant)
The fit is per stand and polarisation over all captures and channels (3 unknowns).
Tests: (1) cross-validation, fitting on alternate captures and predicting the others;
(2) X / Y agreement of Delta r per stand; (3) size of Delta r.

    python dd_positions.py dd_results_*.npy
"""
import sys

import numpy as np
from astropy.time import Time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
sys.path.insert(0, __file__.rsplit("/", 3)[0])
from streamcal import geometry as G, sky as SK  # noqa: E402

SOL = 299792458.0
rows = []
for fn in sys.argv[1:]:
    rows += list(np.load(fn, allow_pickle=True))
rows = sorted(rows, key=lambda r_: (r_["pol"], r_["t"]))


def sdiff(t):
    names, enu, _, _ = SK.point_sources(Time(t), np.array([38.0]), include_sun=False)
    e = dict(zip(names, enu))
    return e["CygA"] - e["CasA"]


fits = {}
for pol in ("XX", "YY"):
    rp = [r_ for r_ in rows if r_["pol"] == pol and tuple(r_["pair"]) in (("CygA", "CasA"), ("CasA", "CygA"))]
    if len(rp) < 4:
        continue
    # design per stand: rows over (capture, channel)
    A_list, Y_list, cap = [], [], []
    for i, r_ in enumerate(rp):
        sgn = 1.0 if tuple(r_["pair"]) == ("CygA", "CasA") else -1.0
        ds = sdiff(r_["t"]) * sgn
        ph = np.imag(r_["dd"])  # [chan, ant]
        for c, f in enumerate(r_["freq"]):
            A_list.append(2 * np.pi * f / SOL * ds)
            Y_list.append(ph[c])
            cap.append(i)
    A = np.array(A_list)  # [rows, 3]
    Y = np.array(Y_list)  # [rows, ant]
    cap = np.array(cap)
    nant = Y.shape[1]
    dr = np.full((nant, 3), np.nan)
    ev_in, ev_cv, tot = [], [], []
    for a in range(nant):
        y = Y[:, a]
        ok = np.isfinite(y)
        if ok.sum() < 10:
            continue
        dr[a] = np.linalg.lstsq(A[ok], y[ok], rcond=None)[0]
        res = y[ok] - A[ok] @ dr[a]
        ev_in.append(1 - np.var(res) / np.var(y[ok]))
        # cross-validation: fit on even captures, predict odd ones, and the reverse
        pred_err, tot_v = 0.0, 0.0
        for train in (0, 1):
            tr = ok & (cap % 2 == train)
            te = ok & (cap % 2 != train)
            if tr.sum() < 6 or te.sum() < 3:
                continue
            x = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
            pred_err += np.sum((y[te] - A[te] @ x) ** 2)
            tot_v += np.sum(y[te] ** 2)
        if tot_v > 0:
            ev_cv.append(1 - pred_err / tot_v)
        tot.append(np.sqrt(np.mean(y[ok] ** 2)))
    fits[pol] = dr
    print(f"{pol}: {len(rp)} captures; DD phase rms {np.degrees(np.median(tot)):.1f} deg; position model explains "
          f"{100 * np.median(ev_in):.0f}% in-sample, {100 * np.median(ev_cv):.0f}% cross-validated (median over stands); "
          f"|Delta r| median {np.nanmedian(np.linalg.norm(dr, axis=1)):.2f} m, per axis rms (E, N, U) "
          f"{np.round(np.sqrt(np.nanmean(dr ** 2, axis=0)), 2).tolist()} m")
if "XX" in fits and "YY" in fits:
    a, b = fits["XX"], fits["YY"]
    ok = np.all(np.isfinite(a), 1) & np.all(np.isfinite(b), 1)
    for k, ax in enumerate("ENU"):
        print(f"X/Y agreement of Delta r_{ax}: correlation {np.corrcoef(a[ok, k], b[ok, k])[0, 1]:+.2f}")
np.save("dd_positions.npy", fits, allow_pickle=True)
