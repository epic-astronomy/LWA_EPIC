"""
Flux density and fluence of candidates, from the calibration of the images.

The monitor works on normalised values (deviations in units of each pixel's
noise), and the images themselves have no absolute scale: the gain tables of
the streaming calibrator are relative (median |c| = 1 per channel and
polarisation) and epic++ applies no primary beam. What the monitor needs
from the calibrator with every gain table is set out in
docs/skymon/streamcal-interface.md: the absolute scale the table normalised away (F_X, F_Y
per channel, data counts per Jy^1/2) and the calibrators of the solution
(position, model spectrum, solved flux scale), and, once the calibrator
calibrates it, the primary beam. Until then the beam is the monitor's own:
the LWA dipole response from lsl (beam_table, model "empirical" by default),
computed once per band; a beam table in the record takes precedence.

With the table applied, Stokes I = XX + YY of an unpolarised source of S Jy
in direction s appears in the image as

    I(s) = k S Beff(s),   Beff(s) = (F_X^2 B_X(s) + F_Y^2 B_Y(s)) / 2,

where k (image units per counts^2) collects the gridding, the transform, the
accumulation over the image and the quantisation. The monitor measures k for
each channel from the calibrators in its own images (measure_calibrators):
the excess, in the 10 s accumulation, of the pixel that sees most of the
calibrator over a quadratic background fitted to a ring around it, divided
by S Beff at the calibrator and by R, the pixel's modelled response to the
calibrator at its known offset (the beam through the pixel's interpolation
taps). A burst's fluence per channel then follows from its signal along the
dispersion track (quality.channel_contributions with the tiers' high-pass
undone, in normalised units), the pixel's noise in image units (the square
root of the running variance), a peak correction P and k Beff at the burst's
position:

    fluence_c = sum_track(x) sigma_c sqrt(2^tier) dt_image P_c / (k_c Beff_c(s))

(a sample of tier j is the sum of 2^j normalised images divided by
sqrt(2^j)). P (peak_fit) is the peak of a Gaussian of the beam's width fitted
to the candidate's pixel and its 8 neighbours, over the pixel's value: a
burst between pixel centres loses up to 17% of its peak (3.6% on average at
nside 64), and the calibrator gets the corresponding correction through R,
so the loss cancels. The error of the fluence follows from the noise of each
channel's contribution (quality.contribution_noise). Nothing about the
imager's internal normalisation is assumed.

Measured on the simulated sky of tests/python/test_flux.py (57 MHz, 64
channels, nside 64): k = 1.00 for Cyg A; over 32 bursts of S/N 40 at pixel
centres, edges, vertices and random positions the fluence is 1.013 +- 0.009
of the truth; 16 bursts of S/N 120 give 1.008 +- 0.005 and 15 bursts at DM
120 give 1.013 +- 0.012. Without the corrections, a burst was measured at
0.82 of its fluence (tier high-pass 9%, pixel offset and interpolation 5%,
noise 6%), and k = 0.98 was the product of two errors that cancelled by
chance (-6.8% at the pixel, +5.3% from a median background).

The record format is the interface with the calibration, which is still
being developed (direction-dependent calibration). The fields used are those
of docs/skymon/streamcal-interface.md; a record without them, or a failure of
any kind, gives candidates without a flux and is logged, and never stops the
search. The record of the simulator (simulate.py) carries its own beam table,
which therefore replaces the lsl model in the simulated tests.

Limits: the peak correction and R assume a Gaussian synthesised beam of
FWHM lambda / D in the image's direction cosines; the real beam's shape and
sidelobes, and direction-dependent calibration, change both. The ring
background holds diffuse emission and sidelobes of other sources that a
quadratic surface describes only in part; the model spectra of the
calibrators are corrected only by the scale the calibrator solves; the dipole
response is a model. Without a calibration record (no streamcal, or its
record missing) the monitor still stores the calibrators' excess in image
units, but no flux.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .geometry import altaz, beam_fwhm_deg


class CalibrationFeed:
    """The calibration records of streamcal (one JSON file, rewritten with every solution)"""

    def __init__(self, path: str | None, keep: int = 2000):
        self.path = path
        self.keep = keep
        self.records = {}  # gain_version -> record (the latest record of each version)
        self.latest = None
        self.unwritten = []  # records for the node's database
        self._stamp = None
        self.errors = 0
        self.last_error = None

    def poll(self):
        """Read the file if it changed; returns the new record or None"""
        if not self.path:
            return None
        try:
            st = os.stat(self.path)
        except OSError:
            return None
        stamp = (st.st_mtime_ns, st.st_size)
        if stamp == self._stamp:
            return None
        try:
            with open(self.path) as f:
                rec = json.load(f)
        except (OSError, ValueError) as e:  # being replaced: the next poll reads it
            self.errors += 1
            self.last_error = f"{type(e).__name__}: {e}"[:200]
            return None
        self._stamp = stamp
        self.add(rec)
        return rec

    def add(self, rec: dict):
        v = int(rec.get("gain_version", 0))
        if self.latest is not None and rec.get("t_solved_unix") == self.latest.get("t_solved_unix") \
                and v == self.latest.get("gain_version"):
            return
        self.records[v] = rec
        self.latest = rec
        self.unwritten.append(rec)
        if len(self.records) > self.keep:
            for k in sorted(self.records)[:len(self.records) - self.keep]:
                del self.records[k]
        del self.unwritten[:-self.keep]

    def for_version(self, version: int | None):
        """The record of the table the images were made with (the latest if unknown or nominal)"""
        if version and version in self.records:
            return self.records[version]
        return self.latest

    def status(self):
        r = self.latest
        return dict(path=self.path, records=len(self.records), errors=self.errors,
                    last_error=self.last_error,
                    latest=None if r is None else dict(
                        gain_version=r.get("gain_version"), applied=r.get("applied"),
                        t_solved_unix=r.get("t_solved_unix"),
                        calibrators=[c["name"] for c in r.get("calibrators", [])],
                        chi2_median=(r.get("quality") or {}).get("chi2_median")))


# ------------------------------------------------------------------ the record
def channel_values(rec_freqs, values, freqs_mhz):
    """Interpolate per-channel values given at rec_freqs (MHz) to freqs_mhz; NaN-safe"""
    x = np.asarray(rec_freqs, float)
    y = np.array([np.nan if v is None else v for v in values], float)
    ok = np.isfinite(y)
    if not ok.any():
        return np.full(len(freqs_mhz), np.nan)
    return np.interp(freqs_mhz, x[ok], y[ok])


def record_freqs(rec):
    """Centre frequencies (MHz) of the record's channels and of its solved channels"""
    from .geometry import CHAN_BW

    allf = (rec["chan0"] + np.arange(rec["nchan"])) * CHAN_BW / 1e6
    solved = np.asarray(rec.get("solve_chans", []), float) * CHAN_BW / 1e6
    return allf, solved


def flux_scales(rec, freqs_mhz):
    """F_X^2 and F_Y^2 per channel (1 if the record has none: nominal images)"""
    if rec is None or not rec.get("applied", False):
        return np.ones(len(freqs_mhz)), np.ones(len(freqs_mhz))
    allf, _ = record_freqs(rec)
    out = []
    for key in ("X", "Y"):
        f = channel_values(allf, rec["flux_scale"][key], freqs_mhz)
        f = np.where(np.isfinite(f), f, np.nanmedian(f) if np.isfinite(f).any() else 1.0)
        out.append(f ** 2)
    return out[0], out[1]


def beam_table(freqs_mhz, model="empirical", alt_step=2.5, az_step=5.0):
    """
    The dipole power response of LWA-SV (lsl.sim.beam, model "empirical" by
    default) on a grid of elevation and azimuth at the given frequencies,
    normalised to the zenith: {model, freqs_mhz, alt_deg, az_deg, X, Y} with
    X, Y [freq][alt][az]. The monitor computes it once per band.
    """
    from lsl.sim.beam import beam_response

    alt = np.arange(0.0, 90.0 + 1e-9, alt_step)
    az = np.arange(0.0, 360.0, az_step)
    A, Z = np.meshgrid(alt, az, indexing="ij")
    out = dict(model=model, freqs_mhz=[float(f) for f in freqs_mhz], alt_deg=alt.tolist(),
               az_deg=az.tolist(), normalised="zenith")
    for pol, key in (("XX", "X"), ("YY", "Y")):
        rows = []
        for f in freqs_mhz:
            b = np.asarray(beam_response(model, pol, Z.ravel(), A.ravel(), frequency=f * 1e6), float)
            b0 = float(np.asarray(beam_response(model, pol, np.array([0.0]), np.array([90.0]),
                                                frequency=f * 1e6), float)[0])
            rows.append((b / b0 if b0 > 0 else b).reshape(A.shape).tolist())
        out[key] = rows
    return out


def _table(rec, tab):
    """
    The beam table to use: the calibration record's when it carries one (the
    calibrator's beam, calibrated after direction-dependent calibration),
    else the monitor's own model (beam_table). Only ratios of the table enter
    the fluence (calibrator against candidate), so its normalisation is free.
    """
    b = (rec or {}).get("beam")
    if b:
        return b
    return tab or None


def beam(tab, el_deg, az_deg, freqs_mhz):
    """B_X, B_Y towards (el, az) per channel, from a beam table (zenith = 1)"""
    if not tab:
        return None
    b = tab
    alt = np.asarray(b["alt_deg"], float)
    az = np.asarray(b["az_deg"], float)
    bf = np.asarray(b["freqs_mhz"], float)
    out = []
    for key in ("X", "Y"):
        t = np.asarray(b[key], float)  # [freq][alt][az]
        vals = np.array([_bilinear(t[k], alt, az, el_deg, az_deg) for k in range(len(bf))])
        out.append(np.interp(freqs_mhz, bf, vals))
    return out[0], out[1]


def _bilinear(t, alt, az, el, a):
    el = float(np.clip(el, alt[0], alt[-1]))
    a = float(a) % 360.0
    i = int(np.clip(np.searchsorted(alt, el) - 1, 0, len(alt) - 2))
    fa = (el - alt[i]) / (alt[i + 1] - alt[i])
    step = 360.0 / len(az)
    j = int(math.floor((a - az[0]) / step)) % len(az)
    j1 = (j + 1) % len(az)
    fz = ((a - az[0]) / step) % 1.0
    v0 = t[i, j] * (1 - fz) + t[i, j1] * fz
    v1 = t[i + 1, j] * (1 - fz) + t[i + 1, j1] * fz
    return v0 * (1 - fa) + v1 * fa


def beam_eff(rec, el_deg, az_deg, freqs_mhz, tab=None):
    """Beff = (F_X^2 B_X + F_Y^2 B_Y) / 2 per channel, or None without a beam table"""
    bxy = beam(_table(rec, tab), el_deg, az_deg, freqs_mhz)
    if bxy is None:
        return None
    fx2, fy2 = flux_scales(rec, freqs_mhz)
    return 0.5 * (fx2 * bxy[0] + fy2 * bxy[1])


# ------------------------------------------------------------------ the beam's peak
def beam_sigma_pix(freqs_mhz, grid_res_deg):
    """Gaussian sigma (image pixels) of the synthesised beam, lambda / D FWHM, per frequency"""
    return np.array([beam_fwhm_deg(float(f)) for f in np.atleast_1d(freqs_mhz)]) / grid_res_deg / 2.3548


def peak_fit(xy, values, sigma_pix, max_factor=1.3, min_factor=0.8, max_offset_pix=1.5):
    """
    The peak of a point source from the values of a HEALPix pixel (first) and
    its neighbours: a Gaussian of the beam's width sigma_pix (image pixels) is
    fitted to values at the pixels' image positions xy [n, 2] (amplitude and
    position). The images are in direction cosines, where the synthesised
    beam is round whatever the elevation, so the fit is made there.

    Returns dict(factor, x, y, applied, reason): factor is the fitted peak over
    the first pixel's value (1 when not applied). It is not applied with fewer
    than 7 pixels, a first value <= 0, a failed fit, a peak more than
    max_offset_pix from the first pixel, or a factor outside
    [min_factor, max_factor].
    """
    xy = np.asarray(xy, float)
    v = np.asarray(values, float)
    out = dict(factor=1.0, x=None, y=None, applied=False, reason=None)
    if len(v) < 7 or not np.all(np.isfinite(v)) or not np.all(np.isfinite(xy)):
        out["reason"] = "neighbours missing"
        return out
    if v[0] <= 0:
        out["reason"] = "no signal at the pixel"
        return out
    s2 = 2.0 * float(sigma_pix) ** 2
    try:
        from scipy.optimize import least_squares

        def resid(p):
            return p[0] * np.exp(-((xy[:, 0] - p[1]) ** 2 + (xy[:, 1] - p[2]) ** 2) / s2) - v

        sol = least_squares(resid, np.array([float(v.max()), xy[0, 0], xy[0, 1]]), method="lm")
        a, px, py = (float(q) for q in sol.x)
    except Exception as e:  # noqa: BLE001 (a failed fit leaves the pixel's value)
        out["reason"] = f"fit failed: {type(e).__name__}"
        return out
    out.update(x=round(px, 4), y=round(py, 4))
    if not np.isfinite(a) or math.hypot(px - xy[0, 0], py - xy[0, 1]) > max_offset_pix:
        out["reason"] = "peak too far from the pixel"
        return out
    f = a / v[0]
    if not (min_factor <= f <= max_factor):
        out["reason"] = f"factor {f:.3f} out of bounds"
        return out
    out.update(factor=round(f, 5), applied=True)
    return out


def peak_factors(factor, freqs_mhz):
    """
    Per-channel peak correction from the band's: the Gaussian beam's width
    scales as 1 / f, so the loss at a fixed offset is exp(-d^2 / 2 sigma_c^2)
    and ln(factor_c) = ln(factor) (f_c / f_mid)^2.
    """
    f = np.asarray(freqs_mhz, float)
    return np.exp(math.log(max(factor, 1e-6)) * (f / float(np.mean(f))) ** 2)


def _model_response(grid, slot, x_src, y_src, freqs_mhz):
    """
    The regridded value at a tracked pixel of a point source of peak 1 at image
    position (x_src, y_src) (1-based FITS), per channel: the Gaussian beam
    sampled at the pixel's interpolation taps. Positions at the grid's last
    update.
    """
    from .geometry import bicubic

    G = grid.grid_size
    xp, yp = grid.xy_a[slot]
    idx, w = bicubic(np.array([xp]), np.array([yp]), G)
    iy, ix = np.divmod(idx[0].astype(np.int64), G)
    sig = beam_sigma_pix(freqs_mhz, grid.grid_res)
    d2 = (ix - (x_src - 1.0)) ** 2 + (iy - (y_src - 1.0)) ** 2
    return np.exp(-d2[None, :] / (2 * sig[:, None] ** 2)) @ w[0].astype(float)


def _ring_background(grid, ring, values, centre_pix, at_pix):
    """
    Background at pixel at_pix from ring values [n, C]: a quadratic surface
    in the tangent plane at centre_pix, fitted with one pass of rejection
    beyond 4 robust deviations (other sources in the ring). A median is
    biased where the background is curved (a calibrator on the Galactic
    ridge: +5% of Cyg A in the simulated sky; quadratic +0.5%). Falls back to
    the median with fewer than 12 ring pixels. Returns (background [C], spread [C]).
    """
    import healpy as hp

    vals = np.asarray(values, float)
    med = np.median(vals, axis=0)
    spread = 1.4826 * np.median(np.abs(vals - med), axis=0)
    if len(ring) < 12:
        return med, spread
    c0 = np.array(hp.pix2vec(grid.nside, int(centre_pix)))
    e1 = np.cross([0.0, 0.0, 1.0], c0)
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c0, e1)

    def design(pix):
        v = np.array(hp.pix2vec(grid.nside, np.asarray(pix, int))).T.reshape(-1, 3)
        u, w = v @ e1, v @ e2
        return np.stack([np.ones_like(u), u, w, u * u, u * w, w * w], 1)

    X = design(ring)
    keep = np.ones(len(ring), bool)
    coef = None
    for _ in range(2):
        coef, *_ = np.linalg.lstsq(X[keep], vals[keep], rcond=None)
        res = vals - X @ coef
        sd = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep], axis=0)), axis=0)
        bad = np.any(np.abs(res) > 4 * np.maximum(sd, 1e-30), axis=1)
        if bad.sum() == 0 or (~bad).sum() < 12:
            break
        keep = ~bad
    res = vals[keep] - X[keep] @ coef
    spread = 1.4826 * np.median(np.abs(res - np.median(res, axis=0)), axis=0)
    return (design([at_pix]) @ coef)[0], spread


# ------------------------------------------------------------------ calibrators in the images
def measure_calibrators(grid, accum, beam_deg, rec, freqs_mhz, unix, window_s=60.0,
                        min_el=20.0, min_snr=5.0, beam_tab=None):
    """
    The calibrators of rec in the monitor's own images: per calibrator, the
    excess (image units) of its pixel over the background, averaged over the
    last window_s of the accumulation, and
    k = excess / (S Beff R), R being the pixel's modelled response to the
    calibrator (_model_response: the beam at the calibrator's known offset
    from the pixel, through the pixel's interpolation taps). The pixel is the
    one, among the calibrator's HEALPix pixel and its neighbours, with the
    largest R; the background is a quadratic surface fitted to a ring of 2-4
    beams (_ring_background), evaluated at that pixel. Candidates get the
    same correction for their own offset (peak_fit), so the beam's loss
    between pixel centres cancels. Returns a dict with the calibrators
    measured and k per channel (the median over calibrators with
    S/N >= min_snr), or None.
    """
    import healpy as hp

    if rec is None or not rec.get("calibrators"):
        return None
    allf, solved = record_freqs(rec)
    out, ks = [], []
    for c in rec["calibrators"]:
        ra, dec = c["ra_deg"], c["dec_deg"]
        el, az = altaz(ra, dec, unix)
        el, az = float(el), float(az)
        if el < min_el:
            continue
        nside = grid.nside
        th, ph = math.radians(90 - dec), math.radians(ra)
        hpix = hp.ang2pix(nside, th, ph)
        cand = [hpix] + [int(p) for p in hp.get_all_neighbours(nside, th, ph) if p >= 0]
        vec = hp.ang2vec(th, ph)
        outer = hp.query_disc(nside, vec, math.radians(4 * beam_deg))
        inner = set(hp.query_disc(nside, vec, math.radians(2 * beam_deg)).tolist())
        ring = [int(p) for p in outer if int(p) not in inner]
        slots = [int(grid.slot_of[p]) for p in cand]
        rslots = [int(grid.slot_of[p]) for p in ring]
        rslots = [s for s in rslots if s >= 0]
        if all(s < 0 for s in slots) or len(rslots) < 8:
            continue
        rings, rpix = [], []
        for p in ring:
            sl = int(grid.slot_of[p])
            if sl < 0:
                continue
            _, d = accum.pixel(sl, since=unix - window_s)
            d = np.asarray(d, float)
            if d.size:
                rings.append(np.nanmean(d, axis=0))
                rpix.append(p)
        if len(rings) < 8:
            continue
        # the pixel that sees most of the calibrator, from its known position
        xs, ys = grid.wcs.all_world2pix(ra, dec, 1, ra_dec_order=True)
        resp, best = None, None
        for p, sl in zip(cand, slots):
            if sl < 0 or not np.all(np.isfinite(grid.xy_a[sl])):
                continue
            r = _model_response(grid, sl, float(xs), float(ys), freqs_mhz)
            if resp is None or float(np.mean(r)) > float(np.mean(resp)):
                resp, best = r, (p, sl)
        if best is None:
            continue
        _, d = accum.pixel(best[1], since=unix - window_s)
        d = np.asarray(d, float)
        if d.size == 0:
            continue
        peak = np.nanmean(d, axis=0)
        bg, spread = _ring_background(grid, rpix, np.array(rings), hpix, best[0])
        excess = (peak - bg) / np.clip(resp, 0.3, 1.2)
        snr = float(np.nansum(peak - bg) / math.sqrt(np.nansum(spread ** 2))) \
            if np.nansum(spread ** 2) > 0 else float("nan")
        model = channel_values(solved, c["model_jy"], freqs_mhz) * (c.get("scale") or 1.0)
        be = beam_eff(rec, el, az, freqs_mhz, beam_tab)
        if be is None:
            continue
        k = excess / np.maximum(model * be, 1e-30)
        out.append(dict(name=c["name"], el_deg=round(el, 2), az_deg=round(az, 2),
                        excess_snr=round(snr, 1), scale=c.get("scale"), hpix=int(best[0]),
                        pixel_response=round(float(np.mean(resp)), 4),
                        background="quadratic" if len(rpix) >= 12 else "median",
                        excess=np.round(excess, 6).tolist(), k=np.round(k, 9).tolist()))
        if np.isfinite(snr) and snr >= min_snr:
            ks.append(k)
    if not out:
        return None
    k = np.nanmedian(np.array(ks), axis=0) if ks else None
    return dict(t=unix, gain_version=rec.get("gain_version"), window_s=window_s,
                calibrators=out, n_used=len(ks),
                k=None if k is None else np.round(k, 9).tolist())


# ------------------------------------------------------------------ candidates
ERR_FACTOR = 1.7  # fluence error / (fluence / S/N) when the window's noise is not given


def candidate_flux(contrib, raw_sigma, tier_level, dt_image, width_s, k, rec, el_deg, az_deg,
                   freqs_mhz, snr=None, beam_tab=None, contrib_err=None, peak=None):
    """
    Fluence (Jy ms) per channel and over the band, and the mean flux density
    over the boxcar (Jy), of a candidate. contrib [C]: its signal along the
    dispersion track (normalised tier units, quality.channel_contributions with
    the high-pass undone); raw_sigma [C]: its pixel's noise per image in image
    units; k [C]: image units per (Jy counts^2), from measure_calibrators.
    contrib_err [C]: the noise of contrib (quality.contribution_noise); the
    fluence error follows from it, or is ERR_FACTOR x fluence / S/N without it
    (the contribution spans more samples than the boxcar of the S/N: 1.66 /
    S/N measured on simulated bursts). peak: peak_fit's result for the
    candidate's pixel and its neighbours; its factor (per channel,
    peak_factors) corrects the fluence for the burst's offset from the pixel.
    Returns None without k or a beam.
    """
    if k is None or rec is None:
        return None
    tab = _table(rec, beam_tab)
    be = beam_eff(rec, el_deg, az_deg, freqs_mhz, tab)
    if be is None:
        return None
    k = np.asarray(k, float)
    img = np.asarray(contrib, float) * np.asarray(raw_sigma, float) * math.sqrt(2 ** tier_level) * dt_image
    den = k * be
    pf = peak_factors(peak["factor"], freqs_mhz) if peak and peak.get("applied") else \
        np.ones(len(den))
    ok = np.isfinite(img) & np.isfinite(den) & (den > 0)
    scale = np.where(ok, pf * np.asarray(raw_sigma, float) * math.sqrt(2 ** tier_level) * dt_image
                     / np.where(ok, den, 1.0), np.nan) * 1e3  # Jy ms per normalised unit
    fl = np.where(ok, np.asarray(contrib, float) * scale, np.nan)
    if not ok.any():
        return None
    band = float(np.nanmean(fl))
    if contrib_err is not None:
        e = np.asarray(contrib_err, float) * scale
        e = e[ok & np.isfinite(e)]
        err = float(math.sqrt(np.sum(e ** 2)) / max(int(ok.sum()), 1)) if len(e) else None
        err_method = "window noise"
    else:
        err = None if not snr else ERR_FACTOR * abs(band) / float(snr)
        err_method = f"{ERR_FACTOR} x fluence / S/N"
    bz = beam(tab, 90.0, 0.0, freqs_mhz)
    rel = beam(tab, el_deg, az_deg, freqs_mhz)
    beam_rel = float(np.nanmean(0.5 * (rel[0] + rel[1]) / (0.5 * (bz[0] + bz[1]))))
    out = dict(fluence_jy_ms=round(band, 3),
               fluence_err_jy_ms=None if err is None else round(err, 3),
               fluence_err_method=err_method,
               flux_density_jy=round(band / 1e3 / max(width_s, 1e-6), 3),
               fluence_per_channel_jy_ms=[None if not np.isfinite(v) else round(float(v), 3) for v in fl],
               # per channel, Jy ms per normalised tier unit of contrib (for burstfit.physical)
               jy_ms_per_unit=[None if not np.isfinite(v) else float(v) for v in scale],
               beam_relative=round(beam_rel, 4), beam_model=tab.get("model"),
               beam_source=tab.get("source", "model"), beam_version=tab.get("version"),
               gain_version=rec.get("gain_version"),
               method="calibrators in the images (flux.py)")
    if peak is not None:
        out["peak_correction"] = dict(peak)
    return out
