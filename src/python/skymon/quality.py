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
  n_eff_channels  the effective number of channels that carry the signal,
                  (sum c)^2 / sum c^2 over the positive contributions c of the
                  channels along the dispersion track (each channel's samples
                  summed over the boxcar at its own delay, channel_contributions). A dispersed burst
                  spreads over most of the band, so this is a large fraction of
                  the channels (even pure noise gives about a third of them);
                  a carrier of an aircraft or of meteor scatter occupies one or
                  two channels of 24 kHz and gives 1-2
  top2_fraction   the share of the positive sum in the two strongest channels:
                  near 1 for narrowband interference, about 2 / nchan for a
                  burst of flat spectrum
"""
from __future__ import annotations

import numpy as np


def _robust_std(x):
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return np.nan
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def track_lags(edges_mhz, row: int) -> np.ndarray:
    """
    For each channel, the samples by which its arrival precedes the arrival at
    the bottom edge of the band, on the dispersion track of delay row (the
    delay of the whole band in samples): proportional to f_c^-2.
    """
    inv2 = np.asarray(edges_mhz, float) ** -2
    fc2 = (0.5 * (np.asarray(edges_mhz[:-1]) + np.asarray(edges_mhz[1:]))) ** -2
    return np.rint(row * (inv2[0] - fc2) / (inv2[0] - inv2[-1])).astype(int)


def channel_contributions(win: np.ndarray, edges_mhz, row: int, t: int, width: int,
                          slack: int = 1) -> np.ndarray:
    """
    Each channel's contribution to the dedispersed boxcar: its samples summed
    over the boxcar of the given width at that channel's delay on the track of
    row. t is the window index of the boxcar's last sample at the bottom edge
    of the band. A channel receives the burst while the sweep crosses it,
    from the arrival at its upper edge to the arrival at its lower edge, so its
    boxcar spans that interval plus the width. The FDMT follows the track only
    to within a sample, so slack samples are added on each side. Samples
    outside the window count as zero.
    """
    C, W = win.shape
    inv2 = np.asarray(edges_mhz, float) ** -2
    lag_e = row * (inv2[0] - inv2) / (inv2[0] - inv2[-1])  # samples before t, per edge
    out = np.zeros(C)
    for c in range(C):
        hi = min(t - int(np.floor(lag_e[c])) + 1 + slack, W)
        lo = max(t - int(np.ceil(lag_e[c + 1])) - width + 1 - slack, 0)
        if hi > lo:
            out[c] = float(np.sum(win[c, lo:hi]))
    return out


def spread_measures(contrib: np.ndarray) -> dict:
    """n_eff_channels and top2_fraction of per-channel contributions (positive part)"""
    pos = np.asarray(contrib, float)
    pos = pos[np.isfinite(pos) & (pos > 0)]
    total = float(pos.sum())
    if total <= 0:
        return dict(n_eff_channels=None, top2_fraction=None)
    top2 = float(np.sum(np.sort(pos)[-2:]))
    return dict(n_eff_channels=round(total ** 2 / float(np.sum(pos ** 2)), 2),
                top2_fraction=round(top2 / total, 3))


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
    lag = track_lags(edges_mhz, row)  # samples before t
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
    # how many channels carry the signal (narrowband interference against a burst)
    out.update(spread_measures(channel_contributions(win, edges_mhz, row, t, width)))
    return out
