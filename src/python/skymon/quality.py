"""
Measures of a candidate within its own band, for review.

A burst found in one band only must be judged on what that band shows. For
the candidate's pixel these numbers are computed from the window that the
tier searched:

  dm_curve        S/N against trial DM at the candidate's time and width
  snr_dm0_ratio   S/N at the lowest trial DM over the best S/N: near 1 for
                  broadband interference, small for a dispersed burst
  subband_snr     S/N in each quarter of the band along the dispersion track:
                  a burst shows in several quarters (unless its spectrum is
                  narrow), narrowband interference in one
  peak_rows       how many trial DMs reach half the best S/N (a measure of
                  how well the DM is determined)
"""
from __future__ import annotations

import numpy as np


def _robust_std(x):
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return np.nan
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def candidate_quality(win: np.ndarray, dmt: np.ndarray, sigma: np.ndarray, edges_mhz,
                      row: int, t: int, width: int, widths, nquarter: int = 4) -> dict:
    """
    win [C, W]: the pixel's tier window; dmt [D + 1, W]: its FDMT; sigma [D + 1, nw]:
    the tier's noise per row and width; row, t: the candidate's delay row and the
    window index of its boxcar's last sample; width: boxcar width in samples.
    """
    C, W = win.shape
    D = dmt.shape[0] - 1
    wi = int(np.nonzero(np.asarray(widths) == width)[0][0]) if width in widths else 0
    cs = np.concatenate([np.zeros((D + 1, 1)), np.cumsum(dmt, axis=1)], axis=1)
    lo = max(t + 1 - width, 0)
    box = cs[:, t + 1] - cs[:, lo]
    s = sigma[:, wi]
    curve = np.where(s > 0, box / np.where(s > 0, s, 1), 0.0)
    best = float(curve[row]) if 0 <= row <= D else float(np.max(curve))
    out = dict(dm_curve=[round(float(v), 2) for v in curve],
               snr_dm0_ratio=round(float(curve[0] / best), 3) if best > 0 else None,
               peak_rows=int(np.sum(curve >= 0.5 * best)) if best > 0 else None)
    # S/N in quarters of the band along the track of this row
    inv2 = np.asarray(edges_mhz, float) ** -2
    fc2 = (0.5 * (np.asarray(edges_mhz[:-1]) + np.asarray(edges_mhz[1:]))) ** -2
    lag = np.rint(row * (inv2[0] - fc2) / (inv2[0] - inv2[-1])).astype(int)  # samples before t
    q = []
    for k in range(nquarter):
        ch = np.arange(k * C // nquarter, (k + 1) * C // nquarter)
        series = np.zeros(W)
        for c in ch:  # align each channel on the bottom-edge arrival
            sh = lag[c]
            series[sh:] += win[c, :W - sh] if sh else win[c]
        bc = np.convolve(series, np.ones(width), "full")[:W]
        sd = _robust_std(np.delete(bc, np.arange(max(0, t - 2 * width), min(W, t + 2 * width))))
        q.append(round(float(bc[t] / sd), 2) if sd and sd > 0 else None)
    out["subband_snr"] = q
    return out
