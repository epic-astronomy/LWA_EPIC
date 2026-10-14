"""
Array geometry, phase centre and image coordinate system for LWA-SV EPIC images.

EPIC grids stand (x, y) only, so whatever part of the geometric delay is not
removed by the phasing ends up as an uncorrected w-term. The LWA-SV stands lie
on a plane tilted by ~1.8 deg (normal near az 124.6 deg, el 88.19 deg). Phasing to
that plane's normal and gridding in the plane's own axes leaves only the
+/-0.22 m scatter about the plane as w; phasing to zenith (the original
behaviour) leaves the whole slope, which shifts sources by up to ~1 px at low
elevation. See EPIC Memo 12 follow-up.

The gridding correction (gridding_correction) uses the actual per-stand kernels;
EPIC_GCF_CORRECTION=legacy restores the previous oversampled-average method.

The phase centre is chosen with the environment variable EPIC_PHASE_CENTER:
  "plane"  (default)  fitted normal of the stand plane
  "zenith"            original behaviour, delay = cable - z/c

Image headers are written by apply_wcs(): CRVAL is the phase centre converted
through AltAz (so precession is applied), and a PC matrix carries the angle
between the grid's +y axis and J2000 north at the phase centre.
"""
import os
from functools import lru_cache

import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, FK5, SkyCoord
from lsl.common.stations import lwasv

SOL = 299792458.0
CHAN_BW = 25.0e3

# Stands excluded from imaging in addition to the status mask (see gen_phases_lwasv)
EXCLUDED_STANDS = frozenset([256] + list(range(122, 129)) + [185, 186])


def phase_center_mode():
    mode = os.environ.get("EPIC_PHASE_CENTER", "plane").strip().lower()
    if mode not in ("plane", "zenith"):
        raise ValueError(f"EPIC_PHASE_CENTER must be 'plane' or 'zenith', not {mode!r}")
    return mode


def station_location():
    return EarthLocation(
        lat=lwasv.lat * 180.0 / np.pi * u.deg,
        lon=lwasv.lon * 180.0 / np.pi * u.deg,
        height=lwasv.elev * u.m,
    )


def used_stands():
    """Boolean mask over stands that EPIC images with (both pols good, not excluded)."""
    ants = lwasv.antennas
    nstand = len(ants) // 2
    mask = np.zeros(nstand, dtype=bool)
    for i in range(nstand):
        ax, ay = ants[2 * i], ants[2 * i + 1]
        mask[i] = (
            ax.combined_status >= 33
            and ay.combined_status >= 33
            and ax.stand.id not in EXCLUDED_STANDS
        )
    return mask


def stand_xyz():
    """Stand positions (east, north, up) in metres, X-pol antenna order."""
    return np.array([(a.stand.x, a.stand.y, a.stand.z) for a in lwasv.antennas[::2]], dtype=np.float64)


def _frame_about(p):
    """Rows (e1, e2, p): e2 is local north projected onto the plane normal to p."""
    p = p / np.linalg.norm(p)
    e1 = np.cross(np.array([0.0, 1.0, 0.0]), p)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(p, e1)
    return np.array([e1, e2, p])


@lru_cache(maxsize=4)
def array_frame(mode=None):
    """
    Returns (p, F, w) for the phase-centre mode:
      p  unit vector (east, north, up) of the phase centre
      F  3x3 rotation whose rows are the grid x axis, grid y axis and p
      w  residual w of each stand in metres after phasing to p (median-subtracted over used stands)
    """
    mode = mode or phase_center_mode()
    xyz = stand_xyz()
    used = used_stands()
    if mode == "zenith":
        p = np.array([0.0, 0.0, 1.0])
        F = np.eye(3)
    else:
        q = xyz[used]
        A = np.c_[q[:, 0], q[:, 1], np.ones(len(q))]
        a, b, _ = np.linalg.lstsq(A, q[:, 2], rcond=None)[0]
        p = np.array([-a, -b, 1.0])
        p /= np.linalg.norm(p)
        F = _frame_about(p)
    w = (xyz @ F.T)[:, 2]
    w = w - np.median(w[used])
    return p, F, w


def phase_center_azel(mode=None):
    p, _, _ = array_frame(mode)
    return np.degrees(np.arctan2(p[0], p[1])) % 360.0, np.degrees(np.arcsin(p[2]))


def _enu_to_altaz(v, obstime, location):
    v = v / np.linalg.norm(v)
    az = np.degrees(np.arctan2(v[0], v[1])) % 360.0
    el = np.degrees(np.arcsin(np.clip(v[2], -1, 1)))
    return SkyCoord(AltAz(az=az * u.deg, alt=el * u.deg, obstime=obstime, location=location))


def wcs_reference(obstime, mode=None):
    """
    J2000 RA/Dec (deg) of the phase centre at obstime, and the rotation (deg) of the
    grid +y axis east of J2000 north at the phase centre (the PC matrix angle).
    """
    p, F, _ = array_frame(mode)
    loc = station_location()
    centre = _enu_to_altaz(p, obstime, loc).transform_to(FK5(equinox="J2000"))
    step = _enu_to_altaz(p + 1e-4 * F[1], obstime, loc).transform_to(FK5(equinox="J2000"))
    rot = centre.position_angle(step).deg
    rot = (rot + 180.0) % 360.0 - 180.0
    return centre.ra.deg, centre.dec.deg, rot


def apply_wcs(header, obstime, grid_size, grid_res, mode=None):
    """
    Write the celestial WCS of an EPIC image into a FITS header.

    Pixel conventions are unchanged from the original writer: east left
    (negative CDELT1), the y axis flipped before saving so CRPIX2 is one lower
    for even grid sizes.
    """
    sll = (2 * grid_size * np.sin(np.pi * grid_res / 360)) ** -1
    dtheta = 2 * np.arcsin(0.5 / (grid_size * sll)) * 180.0 / np.pi
    ra, dec, rot = wcs_reference(obstime, mode)
    th = np.radians(rot)
    header["CTYPE1"] = "RA---SIN"
    header["CRPIX1"] = float(grid_size / 2 + 1)
    header["CDELT1"] = -dtheta
    header["CRVAL1"] = ra
    header["CUNIT1"] = "deg"
    header["CTYPE2"] = "DEC--SIN"
    header["CRPIX2"] = float(grid_size / 2 + 1) - (grid_size + 1) % 2
    header["CDELT2"] = dtheta
    header["CRVAL2"] = dec
    header["CUNIT2"] = "deg"
    header["PC1_1"] = np.cos(th)
    header["PC1_2"] = -np.sin(th)
    header["PC2_1"] = np.sin(th)
    header["PC2_2"] = np.cos(th)
    header["EQUINOX"] = 2000.0
    header["RADESYS"] = "FK5"
    az, el = phase_center_azel(mode)
    header["PCMODE"] = (mode or phase_center_mode(), "EPIC phase centre mode")
    header["PCAZ"] = (az, "phase centre azimuth (deg)")
    header["PCEL"] = (el, "phase centre elevation (deg)")
    return header


def wproj_kernel_weights(gcf_elems, antpos_freq, chan0, nchan, support, grid_size, grid_res, mode=None):
    """
    Fold each stand's residual w-term into its gridding kernel.

    gcf_elems   real kernel weights from ComputeGcfElements, shape (nchan, nstand, support**2)
    antpos_freq grid positions used by the gridder, shape (nchan, nstand, 3), float32
    Returns complex64 weights of the same shape: K_a = FFT[ IFFT(K_a) * exp(-2 pi i w_a (n - 1)) ],
    truncated to the same support x support cells the gridder writes (cell indices follow the
    gridder exactly: dy = half - e / S, dx = e % S - half, pix = int(pos + d) in float32).
    """
    _, _, w_m = array_frame(mode)
    gcf_elems = np.asarray(gcf_elems, dtype=np.float32).reshape(nchan, -1, support * support)
    antpos_freq = np.asarray(antpos_freq, dtype=np.float32).reshape(nchan, -1, 3)
    nstand = gcf_elems.shape[1]
    half = support // 2
    e = np.arange(support * support)
    dy = (half - e // support).astype(np.float32)
    dx = (e % support - half).astype(np.float32)
    dl = 2 * np.sin(np.radians(grid_res) / 2)
    kk = np.fft.fftfreq(grid_size) * grid_size
    r2 = (kk[None, :] * dl) ** 2 + (kk[:, None] * dl) ** 2
    nm1 = np.sqrt(np.clip(1 - r2, 0, 1)) - 1.0
    out = np.zeros(gcf_elems.shape, dtype=np.complex64)
    for c in range(nchan):
        lam = SOL / ((chan0 + c) * CHAN_BW)
        for a in range(nstand):
            k = gcf_elems[c, a]
            if not np.any(k):
                continue
            xp = np.trunc(antpos_freq[c, a, 0] + dx).astype(np.int64)
            yp = np.trunc(antpos_freq[c, a, 1] + dy).astype(np.int64)
            valid = (xp >= 0) & (xp < grid_size) & (yp >= 0) & (yp < grid_size)
            g = np.zeros((grid_size, grid_size), dtype=np.complex128)
            np.add.at(g, (yp[valid], xp[valid]), k[valid])
            g = np.fft.fft2(np.fft.ifft2(g) * np.exp(-2j * np.pi * (w_m[a] / lam) * nm1))
            kw = g[yp[valid], xp[valid]]
            # Truncating to the support loses ~8% of the kernel. Rescale so the
            # response at the phase centre is 1, like the real kernel (sum = 1).
            s0 = np.abs(kw.sum())
            out[c, a, valid] = kw / s0 if s0 > 0 else kw
    return out


def gridding_correction(gcf_elems, antpos_freq, chan0, nchan, support, grid_size, oversample=4):
    """
    Per-pixel correction for the gridding kernel, multiplied into the image by
    BlockEpicImager (gcf_correction_grid).

    The image of a point source is attenuated by |K(l, m)|^2, where K is the mean
    over stands of the Fourier transform of each stand's gridding weights taken
    relative to its true grid position. This includes the one-cell integration in
    GcfPixelIntegral and each stand's sub-cell offset, which the oversampled
    average kernel used before did not (that left the image ~33% too faint at
    30 deg elevation for support 5). W-projection only adds a unit-magnitude
    phase to K, so the same correction applies.

    Returns float32 (nchan, grid_size, grid_size) in the kernel's output order
    ([kx][ky], unshifted). The phase-centre value is (grid_size * oversample)^4,
    as with the previous method, so the absolute scale at the phase centre is
    unchanged.

    EPIC_GCF_CORRECTION=legacy keeps the previous method (handled by the caller).
    """
    S = support
    nel = S * S
    gcf_elems = np.asarray(gcf_elems, dtype=np.float32).reshape(nchan, -1, nel)
    antpos_freq = np.asarray(antpos_freq, dtype=np.float32).reshape(nchan, -1, 3)
    half = S // 2
    e = np.arange(nel)
    dy = (half - e // S).astype(np.float32)
    dx = (e % S - half).astype(np.float32)
    k = (np.fft.fftfreq(grid_size) * grid_size).astype(np.float32)
    scale = float(grid_size * oversample) ** 4
    out = np.empty((nchan, grid_size, grid_size), dtype=np.float32)
    for c in range(nchan):
        w = gcf_elems[c]
        live = np.any(w != 0, axis=1)
        pos = antpos_freq[c, live, :2]
        w = w[live]
        xp = np.trunc(pos[:, 0:1] + dx)
        yp = np.trunc(pos[:, 1:2] + dy)
        ox = (xp - pos[:, 0:1]).ravel()
        oy = (yp - pos[:, 1:2]).ravel()
        wv = w.ravel().astype(np.complex64)
        ex = np.exp(2j * np.pi * np.outer(ox, k) / grid_size).astype(np.complex64)   # J x kx
        ey = np.exp(2j * np.pi * np.outer(oy, k) / grid_size).astype(np.complex64)   # J x ky
        kmean = (ex * wv[:, None]).T @ ey / max(1, int(live.sum()))                    # [kx][ky]
        resp = np.abs(kmean) ** 2
        resp /= resp[0, 0]
        out[c] = scale / np.maximum(resp, 1e-6)
    return out
