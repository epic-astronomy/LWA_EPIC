"""
Physical parameters of a candidate: a model burst fitted to its dynamic spectrum.

The search gives a candidate's DM, time and width on its trial grid. This module
fits the burst model of burst.py, the one the simulator and the injector use,
to the dynamic spectrum the monitor stored with the candidate (the window of
its pixel in the tier that found it, normalised, decimated and high-passed):

  arrival   t_inf + K DM / f^2 at infinite frequency
  shape     a box of the intrinsic width w, swept across each channel,
            convolved with a one-sided exponential of time constant
            tau_1ghz (f / 1 GHz)^-4 (scattering), integrated over each sample
  spectrum  A(f) = A0 (f / f0)^(beta + gamma ln(f / f0)) (a power law with
            curvature; a band-limited burst takes a large negative gamma)

The model samples go through the tier's high-pass, the recursion of
skymon_tier_push, and the window's mean is removed per channel, as the search
did with the data. The parameters are found by non-linear least squares from
the search's values; their uncertainties are the diagonal of the covariance
(J^T J)^-1 scaled by the reduced chi^2.

The amplitude is a fluence in units of the pixel's noise per image times an
image (normalised units), per channel. With the pixel's noise in image units
(raw_sigma) and the calibration (flux.py: k and the beam), the fitted fluences
per channel convert to Jy ms, and the spectral index of the burst follows
from those, independent of the bandpass of the normalisation.
"""
from __future__ import annotations

import math

import numpy as np

from .dedisp import K_DM

# parameters, in the order of the fit
NAMES = ("t_f0", "dm", "log10_width_s", "log10_tau_1ghz_s", "log_amp", "beta", "gamma")


def _highpass(x: np.ndarray, a: float) -> np.ndarray:
    """skymon_tier_push along the last axis: y = x - b, then b += a (x - b), b from 0"""
    if a <= 0:
        return x
    y = np.empty_like(x)
    b = np.zeros(x.shape[:-1])
    for t in range(x.shape[-1]):
        y[..., t] = x[..., t] - b
        b = b + a * (x[..., t] - b)
    return y


def fractions(edges_mhz, dm, t_inf, width_s, tau_1ghz, t_edges):
    """
    Fraction of each channel's fluence in each sample [t_edges[k], t_edges[k+1]),
    [C, len(t_edges) - 1]: burst.channel_fractions for all samples at once.
    """
    e = np.asarray(edges_mhz, float)
    a = (t_inf + K_DM * dm * e[1:] ** -2)[:, None]  # the top edge of each channel arrives first
    b = (t_inf + K_DM * dm * e[:-1] ** -2 + width_s)[:, None]
    L = np.maximum(b - a, 1e-9)
    T = np.asarray(t_edges, float)[None, :]
    if tau_1ghz <= 0:
        F = np.clip(np.minimum(b, T) - np.minimum(a, T), 0.0, None) / L
    else:
        tau = (tau_1ghz * (0.5 * (e[:-1] + e[1:]) / 1000.0) ** -4)[:, None]

        def H(u):
            u = np.maximum(u, 0.0)
            return u - tau * -np.expm1(-u / tau)

        F = (H(T - a) - H(T - b)) / L
    return np.clip(np.diff(F, axis=1), 0.0, 1.0)


def model(p, edges_mhz, t0, dt, nsamp, a_hp=0.0, f0=None, scale=1.0):
    """
    The burst of parameters p (NAMES) in a window of nsamp samples of dt from t0, [C, nsamp].
    The time is the arrival at f0 (the band's geometric centre): with the arrival
    at infinite frequency, time and DM would be almost degenerate.
    """
    t_f0, dm, lw, lt, la, beta, gamma = p
    e = np.asarray(edges_mhz, float)
    fc = 0.5 * (e[:-1] + e[1:])
    f0 = f0 or float(np.sqrt(e[0] * e[-1]))
    t_inf = t_f0 - K_DM * dm / f0 ** 2
    out = fractions(e, float(dm), float(t_inf), 10.0 ** lw, 10.0 ** lt if lt > -12 else 0.0,
                    t0 + dt * np.arange(nsamp + 1))
    x = np.log(fc / f0)
    amp = np.exp(la + beta * x + gamma * x * x)
    out *= (amp * scale)[:, None]
    out = _highpass(out, a_hp)
    return out - out.mean(axis=1, keepdims=True)


def fit(dynspec, times, edges_mhz, dm, t_inf, width_s, tier_images=1, a_hp=0.0, chan_ok=None,
        max_nfev=200, dm_step=None):
    """
    Fit the burst model to dynspec [C, W] (normalised tier samples at times, s).

    dm, t_inf, width_s: the search's values, the starting point; dm_step: the
    search's trial step (the fit starts from the DMs within it). tier_images:
    images per tier sample (the tier's samples are sums of that many images
    over its square root). Returns a dict of the parameters with their
    uncertainties, the fitted fluence per channel (normalised units: pixel
    noise per image x images) and the quality of the fit; None if it failed.
    """
    from scipy.optimize import least_squares

    d = np.asarray(dynspec, np.float64)
    C, W = d.shape
    # times relative to the search's arrival: the solver's tolerances are relative to the
    # parameters, and a Unix time of 1.8e9 s would let it stop many seconds short
    t_ref = float(t_inf)
    t = np.asarray(times, float) - t_ref
    t_inf = 0.0
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    if dt <= 0 or W < 8:
        return None
    ok = np.ones(C, bool) if chan_ok is None else np.asarray(chan_ok, bool)
    ok &= np.isfinite(d).all(axis=1) & (np.abs(d).sum(axis=1) > 0)
    if ok.sum() < 8:
        return None
    e = np.asarray(edges_mhz, float)
    fc = 0.5 * (e[:-1] + e[1:])
    f0 = float(np.sqrt(e[0] * e[-1]))
    # a tier sample is the sum of n images over sqrt(n): a fluence of A (per image noise x image)
    # adds A / sqrt(n) to the samples it falls in
    scale = 1.0 / math.sqrt(max(int(tier_images), 1))
    sweep = K_DM * (e[0] ** -2 - e[-1] ** -2)
    w0 = max(width_s, dt)
    t_f0 = t_inf + K_DM * dm / f0 ** 2
    step = float(dm_step) if dm_step else max(0.5, 0.01 * dm)
    # the width and the scattering at f0 within a few of the search's width: a
    # longer smear trades against the amplitude and drifts off on the noise
    long_s = 6 * max(w0, dt)
    lt_hi = min(-3.0, math.log10(long_s * (f0 / 1000.0) ** 4))
    lo = np.array([t_f0 - max(2 * w0, 4 * dt), max(dm - 4 * step, 0.0), math.log10(dt / 50), -12.0, -10.0, -10, -40])
    hi = np.array([t_f0 + max(2 * w0, 4 * dt), dm + 4 * step, math.log10(max(6 * w0, 4 * dt)), lt_hi, 30.0, 10, 40])

    def resid(p):
        return (model(p, e, t[0], dt, W, a_hp, f0, scale) - d)[ok].ravel()

    def start(dm1):
        # the amplitude that best fits a flat spectrum at this DM
        p = np.array([t_f0, dm1, math.log10(w0), -8.0, 0.0, 0.0, 0.0])
        m1 = model(p, e, t[0], dt, W, a_hp, f0, scale)
        num, den = float(np.sum(d[ok] * m1[ok])), float(np.sum(m1[ok] ** 2))
        p[4] = math.log(max(num / den, 1e-3)) if den > 0 else 0.0
        return np.clip(p, lo + 1e-9, hi - 1e-9)

    # starts across the search's trial step; the best one goes on to the full fit
    best = None
    for dm1 in dm + step * np.array([-0.5, 0.0, 0.5]):
        try:
            r = least_squares(resid, start(dm1), bounds=(lo, hi), x_scale=[dt, step, 0.3, 1.0, 0.5, 1.0, 1.0],
                              max_nfev=max(20, max_nfev // 4))
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    try:
        r = least_squares(resid, best.x, bounds=(lo, hi), x_scale=[dt, step, 0.3, 1.0, 0.5, 1.0, 1.0],
                          max_nfev=max_nfev)
    except Exception:
        r = best
    n, k = r.fun.size, len(r.x)
    chi2 = float(np.sum(r.fun ** 2))
    red = chi2 / max(n - k, 1)
    try:
        cov = np.linalg.inv(r.jac.T @ r.jac) * red
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(k, np.nan)
    p = r.x
    x = np.log(fc / f0)
    amp = np.exp(p[4] + p[5] * x + p[6] * x * x)
    # S/N of the fitted model: its norm over the noise (unit per sample after normalisation)
    m = model(p, e, t[0], dt, W, a_hp, f0, scale)
    snr = float(np.sqrt(np.sum(m[ok] ** 2)))
    tau = 10.0 ** p[3]
    out = dict(
        converged=bool(r.success), message=str(r.message)[:120], nfev=int(r.nfev), chi2_reduced=round(red, 4),
        snr_model=round(snr, 2), f0_mhz=round(f0, 4),
        t_f0=float(t_ref + p[0]), t_f0_err=float(err[0]), t_inf=float(t_ref + p[0] - K_DM * p[1] / f0 ** 2),
        dm=float(p[1]), dm_err=float(err[1]),
        width_s=float(10.0 ** p[2]), width_err_s=float(10.0 ** p[2] * math.log(10) * err[2]),
        tau_1ghz_s=float(tau), tau_1ghz_err_s=float(tau * math.log(10) * err[3]),
        tau_f0_s=float(tau * (f0 / 1000.0) ** -4),
        beta=float(p[5]), beta_err=float(err[5]), gamma=float(p[6]), gamma_err=float(err[6]),
        fluence_norm=[float(v) for v in amp],
        sweep_s=float(sweep * p[1]))
    # at a bound: the fit did not find the burst's parameters there
    out["at_bound"] = [NAMES[i] for i in range(k) if p[i] <= lo[i] + 1e-6 * (hi[i] - lo[i]) or p[i] >= hi[i] - 1e-6 * (hi[i] - lo[i])]
    # reliable: converged with the time, DM and amplitude inside their bounds, and the
    # width and scattering either inside or at their lower bounds (unresolved)
    upper = [NAMES[i] for i in (2, 3) if p[i] >= hi[i] - 1e-6 * (hi[i] - lo[i])]
    out["reliable"] = bool(r.success and not upper and not set(out["at_bound"]) & {"t_f0", "dm", "log_amp"}
                           and np.isfinite(err[:2]).all() and err[1] < 4 * step)
    # unresolved: the width or the scattering below about a fifth of a sample
    out["width_resolved"] = bool(out["width_s"] > 0.2 * dt)
    out["scattering_resolved"] = bool(out["tau_f0_s"] > 0.2 * dt and out["tau_1ghz_err_s"] < out["tau_1ghz_s"])
    return out


def physical(fitres, flux, tier_level, freqs_mhz):
    """
    The fitted burst in physical units with the candidate's calibration (flux.py):
    per channel, flux["jy_ms_per_unit"] Jy ms per normalised tier unit of the
    dispersion track's sum, which is the fitted fluence over sqrt(2^tier). Gives
    the fluence per channel and over the band (Jy ms), the peak flux density
    over the fitted width (Jy), and the spectral index of the fluence (a power
    law fitted to the channels); None without a calibration.
    """
    if not fitres or not flux or not flux.get("jy_ms_per_unit"):
        return None
    sc = np.array([np.nan if v is None else v for v in flux["jy_ms_per_unit"]], float)
    a = np.asarray(fitres["fluence_norm"], float) / math.sqrt(2 ** tier_level) * sc
    good = np.isfinite(a) & (a > 0)
    if not good.any():
        return None
    out = dict(fluence_per_channel_jy_ms=[None if not np.isfinite(v) else round(float(v), 4) for v in a],
               fluence_jy_ms=round(float(np.nanmean(np.where(np.isfinite(a), a, np.nan))), 4))
    w = max(fitres["width_s"], 1e-6)
    out["flux_density_jy"] = round(out["fluence_jy_ms"] / 1e3 / w, 4)
    f = np.asarray(freqs_mhz, float)
    if good.sum() >= 4:
        x, y = np.log(f[good]), np.log(a[good])
        A = np.c_[np.ones_like(x), x - x.mean()]
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        s2 = float(res[0]) / max(len(x) - 2, 1) if len(res) else float("nan")
        cov = s2 * np.linalg.inv(A.T @ A)
        out["spectral_index"] = round(float(coef[1]), 3)
        out["spectral_index_err"] = round(float(math.sqrt(max(cov[1, 1], 0))), 3)
    out["calibration"] = dict(gain_version=flux.get("gain_version"), beam_model=flux.get("beam_model"),
                              method=flux.get("method"))
    return out
