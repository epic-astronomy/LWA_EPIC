"""Position / phase-centre model with a free phase offset per stand and capture.

phi_a(f, t) = 2 pi f / c  Delta r_a . (s_CygA(t) - s_CasA(t)) + c_a(t)
Cross-validation: Delta r_a is fitted on alternate captures (with their offsets); on the
held-out captures only c_a(t) is refitted, and the variance explained beyond an
offset-only model is reported. Also: X / Y agreement of Delta r, its size, and how much a
free per-capture delay (phi = 2 pi f tau_a(t) + c_a(t)) explains (the upper bound for
any delay-like model).
    python dd_positions2.py dd_results_*.npy
"""
import sys

import numpy as np
from astropy.time import Time

sys.path.insert(0, __file__.rsplit("/", 3)[0])
from streamcal import sky as SK  # noqa: E402

SOL = 299792458.0
rows = []
for fn in sys.argv[1:]:
    rows += list(np.load(fn, allow_pickle=True))


def sdiff(t):
    names, enu, _, _ = SK.point_sources(Time(t), np.array([38.0]), include_sun=False)
    e = dict(zip(names, enu))
    return e["CygA"] - e["CasA"]


fits = {}
for pol in ("XX", "YY"):
    rp = sorted([r_ for r_ in rows if r_["pol"] == pol], key=lambda r_: r_["t"])
    rp = [r_ for r_ in rp if len(r_["freq"]) >= 8]
    caps = []
    for r_ in rp:
        sgn = 1.0 if tuple(r_["pair"]) == ("CygA", "CasA") else -1.0
        caps.append((r_["freq"], np.imag(r_["dd"]) * sgn, sdiff(r_["t"])))
    nant = caps[0][1].shape[1]
    ev_delay, ev_pos_cv, dr = [], [], np.full((nant, 3), np.nan)
    for a in range(nant):
        # per-capture delay + offset (upper bound for delay-like models)
        tot, res_d = 0.0, 0.0
        for f, ph, _ in caps:
            y = ph[:, a]
            ok = np.isfinite(y)
            if ok.sum() < 4:
                continue
            X = np.c_[np.ones(ok.sum()), f[ok]]
            c = np.linalg.lstsq(X, y[ok], rcond=None)[0]
            res_d += np.sum((y[ok] - X @ c) ** 2)
            tot += np.sum((y[ok] - y[ok].mean()) ** 2)
        if tot == 0:
            continue
        ev_delay.append(1 - res_d / tot)

        # position model with per-capture offsets, fitted on a subset of captures
        def fit(idx):
            blocks, ys = [], []
            for j in idx:
                f, ph, ds = caps[j]
                y = ph[:, a]
                ok = np.isfinite(y)
                if ok.sum() < 4:
                    continue
                A = 2 * np.pi * f[ok, None] / SOL * ds[None, :]
                Y = y[ok]
                # remove the per-capture offset by centring both sides
                blocks.append(A - A.mean(0))
                ys.append(Y - Y.mean())
            if not blocks:
                return None
            return np.linalg.lstsq(np.vstack(blocks), np.concatenate(ys), rcond=None)[0]

        dr[a] = fit(range(len(caps)))
        num, den = 0.0, 0.0
        for train in (0, 1):
            x = fit([j for j in range(len(caps)) if j % 2 == train])
            if x is None:
                continue
            for j in range(len(caps)):
                if j % 2 == train:
                    continue
                f, ph, ds = caps[j]
                y = ph[:, a]
                ok = np.isfinite(y)
                if ok.sum() < 4:
                    continue
                pred = 2 * np.pi * f[ok] / SOL * (ds @ x)
                yc, pc = y[ok] - y[ok].mean(), pred - pred.mean()
                num += np.sum((yc - pc) ** 2)
                den += np.sum(yc ** 2)
        if den > 0:
            ev_pos_cv.append(1 - num / den)
    fits[pol] = dr
    print(f"{pol}: {len(caps)} captures. Variance about the per-capture offset explained by: a free delay per capture "
          f"{100 * np.median(ev_delay):.0f}% (median over stands); a fixed Delta r per stand, cross-validated "
          f"{100 * np.median(ev_pos_cv):.0f}%. |Delta r| median {np.nanmedian(np.linalg.norm(dr, axis=1)):.2f} m, "
          f"rms (E, N, U) {np.round(np.sqrt(np.nanmean(dr ** 2, 0)), 2).tolist()} m")
if len(fits) == 2:
    a, b = fits["XX"], fits["YY"]
    ok = np.all(np.isfinite(a), 1) & np.all(np.isfinite(b), 1)
    print("X/Y agreement of Delta r (E, N, U):", [f"{np.corrcoef(a[ok, k], b[ok, k])[0, 1]:+.2f}" for k in range(3)])
