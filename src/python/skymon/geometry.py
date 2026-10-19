"""
Sky-fixed pixelisation for the all-sky monitor.

The monitor tracks HEALPix pixels (equatorial, J2000, RING order) instead of
image pixels, so a pixel's time series always looks at the same patch of sky.
Every regrid interval the tracked set is updated: pixels that rose above the
elevation limit are given a slot, pixels that set are released, and each
tracked pixel's position on the image is recomputed with the same celestial
WCS the imager writes into its FITS files (epic_geometry.apply_wcs).

Image arrays follow the FITS data order of the saved images, [chan][y][x],
with FITS pixel (x, y) at array index [y - 1][x - 1].
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import numpy as np

# make the imager's own geometry module importable (src/python)
_PYDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PYDIR not in sys.path:
    sys.path.insert(0, _PYDIR)

FS = 196.0e6
SEQ_MULT_S = 8192
CHAN_BW = FS / SEQ_MULT_S  # NDP channel width, Hz
LWASV_LAT = 34.3484  # deg (lsl lwasv)
LWASV_LON = -106.8858  # deg
STATION_DIAMETER = 100.0  # m, used for the beam width


def seq_to_unix(seq: int) -> float:
    """NDP sequence number (one per spectrum) to Unix time, s"""
    return seq * SEQ_MULT_S / FS


@dataclass(frozen=True)
class Band:
    chan0: int  # absolute NDP channel of the first (binned) channel
    nchan: int  # channels after binning
    nbin: int = 1  # NDP channels per channel

    @property
    def chan_bw_hz(self) -> float:
        return CHAN_BW * self.nbin

    def centres_mhz(self) -> np.ndarray:
        # NDP channel k is centred on k * CHAN_BW (epic_geometry.wproj_kernel_weights)
        k = self.chan0 + self.nbin * np.arange(self.nchan) + (self.nbin - 1) / 2
        return k * CHAN_BW / 1e6

    def edges_mhz(self) -> np.ndarray:
        c = self.centres_mhz()
        return np.r_[c - self.chan_bw_hz / 2e6, c[-1] + self.chan_bw_hz / 2e6]

    @property
    def f_lo(self) -> float:
        return float(self.edges_mhz()[0])

    @property
    def f_hi(self) -> float:
        return float(self.edges_mhz()[-1])

    @property
    def f_centre(self) -> float:
        return 0.5 * (self.f_lo + self.f_hi)


def beam_fwhm_deg(f_mhz: float) -> float:
    """Approximate synthesised beam FWHM of LWA-SV, lambda / D"""
    return math.degrees(299.792458 / f_mhz / STATION_DIAMETER)


def auto_nside(f_mhz: float, image_res_deg: float = 1.056) -> int:
    """Coarsest HEALPix nside whose pixels are at most half the beam (but not finer than the image)"""
    target = max(beam_fwhm_deg(f_mhz) / 2.0, image_res_deg)
    nside = 8
    while math.degrees(math.sqrt(4 * math.pi / (12 * nside ** 2))) > target:
        nside *= 2
    return nside


def lst_deg(unix: float, lon_deg: float = LWASV_LON) -> float:
    """Local mean sidereal time (deg); accurate to a few arcsec, enough for elevation cuts"""
    jd = unix / 86400.0 + 2440587.5
    d = jd - 2451545.0
    gmst = 280.46061837 + 360.98564736629 * d
    return (gmst + lon_deg) % 360.0


def altaz(ra_deg, dec_deg, unix, lat_deg=LWASV_LAT, lon_deg=LWASV_LON):
    """Elevation and azimuth (deg) of J2000 positions; precession ignored (~0.4 deg)"""
    ha = np.radians(lst_deg(unix, lon_deg) - np.asarray(ra_deg))
    dec = np.radians(dec_deg)
    lat = math.radians(lat_deg)
    sin_el = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)
    el = np.arcsin(np.clip(sin_el, -1, 1))
    az = np.arctan2(-np.cos(dec) * np.sin(ha),
                    np.sin(dec) * np.cos(lat) - np.cos(dec) * np.cos(ha) * np.sin(lat))
    return np.degrees(el), np.degrees(az) % 360.0


def time_to_set(dec_deg, ra_deg, unix, el_min_deg, lat_deg=LWASV_LAT, lon_deg=LWASV_LON):
    """Seconds until each position sets below el_min (inf if it never sets, 0 if below)"""
    dec = np.radians(np.asarray(dec_deg, float))
    lat, elm = math.radians(lat_deg), math.radians(el_min_deg)
    c = (math.sin(elm) - math.sin(lat) * np.sin(dec)) / (math.cos(lat) * np.cos(dec))
    h_set = np.degrees(np.arccos(np.clip(c, -1, 1)))
    ha = (lst_deg(unix, lon_deg) - np.asarray(ra_deg) + 180.0) % 360.0 - 180.0
    left = (h_set - ha) / 15.0 * 3600.0 * 0.99727
    left = np.where(c <= -1, np.inf, left)
    return np.where((c >= 1) | (ha > h_set), 0.0, np.maximum(left, 0.0))


def image_wcs(unix: float, grid_size: int, grid_res: float, mode: str | None = None):
    """astropy WCS of an EPIC image starting at unix, identical to the saved FITS"""
    from astropy.io import fits
    from astropy.time import Time
    from astropy.wcs import WCS
    from epic_geometry import apply_wcs

    header = fits.ImageHDU(np.zeros((grid_size, grid_size), np.float32)).header
    apply_wcs(header, Time(unix, format="unix", precision=6), grid_size, grid_res, mode)
    return WCS(header, naxis=2)


def bilinear(x, y, grid_size):
    """
    Bilinear taps for 1-based FITS pixel coordinates on a [y][x] array.
    Returns flat indices [P, 4] (int32) and weights [P, 4] (float32); points
    outside the grid get zero weight.
    """
    x0 = np.floor(np.asarray(x) - 1.0)
    y0 = np.floor(np.asarray(y) - 1.0)
    fx = (np.asarray(x) - 1.0 - x0).astype(np.float32)
    fy = (np.asarray(y) - 1.0 - y0).astype(np.float32)
    ok = np.isfinite(x0) & np.isfinite(y0) & (x0 >= 0) & (y0 >= 0) & \
        (x0 < grid_size - 1) & (y0 < grid_size - 1)
    x0 = np.where(ok, x0, 0).astype(np.int64)
    y0 = np.where(ok, y0, 0).astype(np.int64)
    idx = np.stack([y0 * grid_size + x0, y0 * grid_size + x0 + 1,
                    (y0 + 1) * grid_size + x0, (y0 + 1) * grid_size + x0 + 1], axis=1)
    w = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], axis=1)
    w[~ok] = 0.0
    return idx.astype(np.int32), w.astype(np.float32)


def _keys(f, a=-0.75):
    """
    Keys cubic convolution weights for offsets f in [0, 1), taps at -1, 0, 1, 2.
    a = -0.75 rather than Keys' -0.5: on the beam of the images (about three
    image pixels per FWHM) a = -0.5 underestimates a peak by 2.5% on average
    (up to 4.8%), a = -0.75 reproduces it to within +0.8% (+0.4% on average). The price is that
    a plane is no longer reproduced exactly: a gradient of g per pixel is
    offset by at most 0.048 g, which ripples with the sub-pixel position over
    minutes as the sky turns and is removed, like the slow sky, by the
    running mean, the high-pass and the detrend.
    """
    t = np.stack([1 + f, f, 1 - f, 2 - f], -1)
    t2 = t * t
    t3 = t2 * t  # as kernels.taps computes it (a product, not pow)
    near = (a + 2) * t3 - (a + 3) * t2 + 1
    far = a * t3 - 5 * a * t2 + 8 * a * t - 4 * a
    return np.where(t <= 1, near, far)


def bicubic(x, y, grid_size):
    """
    Bicubic (Keys, a = -0.75) taps for 1-based FITS pixel coordinates on a [y][x]
    array: flat indices [P, 16] (int32) and weights [P, 16] (float32). Points
    within two pixels of the edge, or not finite, get zero weight.

    On a beam sampled with about three image pixels per FWHM, bilinear
    interpolation loses 10% of a peak on average (up to 15%), depending on
    where the position falls between pixels, and that error ripples as the
    sky turns. Keys' a = -0.5 loses 2.5% on average (up to 4.8%); a = -0.75
    reproduces the peak to between 0.0% and +0.8%, +0.4% on average (Gaussian beam,
    1500 random positions; docs/skymon/DESIGN.md section 2.2).
    """
    x = np.asarray(x, float) - 1.0
    y = np.asarray(y, float) - 1.0
    x0, y0 = np.floor(x), np.floor(y)
    ok = np.isfinite(x0) & np.isfinite(y0) & (x0 >= 1) & (y0 >= 1) & \
        (x0 < grid_size - 2) & (y0 < grid_size - 2)
    x0 = np.where(ok, x0, 1).astype(np.int64)
    y0 = np.where(ok, y0, 1).astype(np.int64)
    wx = _keys(np.where(ok, x - x0, 0.0))  # [P, 4]
    wy = _keys(np.where(ok, y - y0, 0.0))
    d = np.arange(-1, 3)
    idx = ((y0[:, None, None] + d[None, :, None]) * grid_size +
           (x0[:, None, None] + d[None, None, :])).reshape(-1, 16)
    w = (wy[:, :, None] * wx[:, None, :]).reshape(-1, 16)
    w[~ok] = 0.0
    return idx.astype(np.int32), w.astype(np.float32)


class SkyGrid:
    """
    HEALPix pixels tracked by the monitor and their slots in the state arrays.

    A pixel is tracked while it is above el_min. Slots are reused: when a pixel
    sets its slot is released and may be given to a pixel that rises later.
    """

    def __init__(self, nside: int, el_min_deg: float = 10.0, grid_size: int = 128,
                 grid_res: float = 1.056, phase_mode: str | None = None):
        import healpy as hp

        self.nside = nside
        self.el_min = el_min_deg
        self.grid_size = grid_size
        self.grid_res = grid_res
        self.phase_mode = phase_mode
        self.npix = hp.nside2npix(nside)
        th, ph = hp.pix2ang(nside, np.arange(self.npix))
        self.ra = np.degrees(ph)
        self.dec = 90.0 - np.degrees(th)
        # every pixel ever above el_min, and the most above it at once (+5%)
        ever = self.dec > LWASV_LAT - 90.0 + el_min_deg
        frac_up = (1 - math.sin(math.radians(el_min_deg))) / 2
        self.capacity = int(math.ceil(self.npix * frac_up * 1.05)) + 16
        self.ever = ever
        self.slot_of = np.full(self.npix, -1, np.int64)
        self.hpix_of = np.full(self.capacity, -1, np.int64)
        # the slots of each slot's 8 HEALPix neighbours (-1: none or not tracked),
        # for the normaliser's test of a sky-shaped cell (skymon_normalise)
        self.nbr = np.full((self.capacity, 8), -1, np.int32)
        self.idx = np.zeros((self.capacity, 16), np.int32)
        self.w = np.zeros((self.capacity, 16), np.float32)
        self.xy = np.full((self.capacity, 2), np.nan)
        self.updated_unix = None

    @property
    def pix_deg(self) -> float:
        return math.degrees(math.sqrt(4 * math.pi / self.npix))

    def active(self) -> np.ndarray:
        return self.hpix_of >= 0

    def update(self, unix: float, interval: float = 10.0):
        """
        Recompute the tracked set for an image starting at unix, and the image
        position of every tracked pixel at unix and at unix + interval; taps()
        interpolates between the two for each image, so a pixel's position
        moves smoothly with the sky instead of jumping at every update.
        Returns (added_slots, removed_slots).
        """
        el, _ = altaz(self.ra, self.dec, unix)
        up = np.nonzero((el > self.el_min) & self.ever)[0]
        wcs = image_wcs(unix, self.grid_size, self.grid_res, self.phase_mode)
        self.wcs = wcs
        x, y = wcs.all_world2pix(self.ra[up], self.dec[up], 1, ra_dec_order=True)
        inside = np.isfinite(x) & np.isfinite(y) & (x >= 1) & (x <= self.grid_size - 1) & \
            (y >= 1) & (y <= self.grid_size - 1)
        up, x, y = up[inside], x[inside], y[inside]

        want = np.zeros(self.npix, bool)
        want[up] = True
        removed = []
        for s in np.nonzero(self.hpix_of >= 0)[0]:
            h = self.hpix_of[s]
            if not want[h]:
                self.slot_of[h] = -1
                self.hpix_of[s] = -1
                removed.append(s)
        free = list(np.nonzero(self.hpix_of < 0)[0][::-1])
        added = []
        for h in up:
            if self.slot_of[h] < 0:
                if not free:
                    break  # over capacity: should not happen; skip the rest
                s = free.pop()
                self.slot_of[h] = s
                self.hpix_of[s] = h
                added.append(s)
        slots = self.slot_of[up]
        keep = slots >= 0
        wcs_b = image_wcs(unix + interval, self.grid_size, self.grid_res, self.phase_mode)
        xb, yb = wcs_b.all_world2pix(self.ra[up], self.dec[up], 1, ra_dec_order=True)
        self.xy_a = np.full((self.capacity, 2), np.nan)
        self.xy_b = np.full((self.capacity, 2), np.nan)
        self.xy_a[slots[keep]] = np.stack([x[keep], y[keep]], 1)
        self.xy_b[slots[keep]] = np.stack([xb[keep], yb[keep]], 1)
        self.t_a, self.t_b = unix, unix + interval
        self.updated_unix = unix
        self._update_neighbours()
        self.taps(unix)
        return np.asarray(added, np.int64), np.asarray(removed, np.int64)

    def _update_neighbours(self):
        import healpy as hp

        self.nbr[:] = -1
        act = np.nonzero(self.hpix_of >= 0)[0]
        if len(act) == 0:
            return
        nb = hp.get_all_neighbours(self.nside, self.hpix_of[act])  # [8, n]
        sl = np.where(nb >= 0, self.slot_of[np.maximum(nb, 0)], -1)
        self.nbr[act] = sl.T.astype(np.int32)

    def taps(self, unix: float, nthreads: int = 1):
        """
        Bicubic taps of every tracked pixel for an image starting at unix
        (kernels.taps: the arithmetic of bicubic, in compiled code, as it runs
        for every image)
        """
        from . import kernels

        f = min(max((unix - self.t_a) / (self.t_b - self.t_a), 0.0), 1.5)
        kernels.taps(self.xy_a, self.xy_b, f, self.grid_size, self.xy, self.idx, self.w, nthreads)

    def nearest_slot(self, ra: float, dec: float) -> int:
        import healpy as hp

        h = hp.ang2pix(self.nside, math.radians(90 - dec), math.radians(ra))
        return int(self.slot_of[h])

    def neighbours(self, hpix: int) -> np.ndarray:
        import healpy as hp

        n = hp.get_all_neighbours(self.nside, int(hpix))
        return n[n >= 0]
