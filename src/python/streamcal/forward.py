"""Forward model: sky -> model visibility matrix per channel (one dipole polarisation).

    M_ab(f) = sum_s B_p(s, f) S_s(f) exp(2 pi i f (r_a - r_b) . s / c)

r are full 3D stand positions (east, north, up), so no w approximation is
made here; EPIC's own w handling is in the kernels (epicop). Computed as
M = A diag(B S) A^H with A[a, s] = exp(2 pi i f r_a . s / c) (a BLAS GEMM).

Convention (matches gen_phases_lwasv): voltage x_a ~ exp(+2 pi i f r_a . s / c),
and the imager's geometric phase is phi_a = exp(-2 pi i f p . r_a / c).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy.time import Time

from . import sky as sm
from . import geometry as E

SOL = 299792458.0


def stand_enu():
    return E.eg.stand_xyz()


def geometric_phase(chan: int, ant: np.ndarray):
    p, _, _ = E.eg.array_frame()
    f = chan * E.CHAN_BW
    r = stand_enu()[ant]
    return np.exp(-2j * np.pi * f * (r @ p) / SOL)


def dipole_beam(enu, freq_hz, pol="XX", model="empirical"):
    from lsl.sim.beam import beam_response

    alt = np.degrees(np.arcsin(np.clip(enu[:, 2], -1, 1)))
    az = np.degrees(np.arctan2(enu[:, 0], enu[:, 1])) % 360
    return np.asarray(beam_response(model, pol, az, alt, frequency=freq_hz), float)


@dataclass
class Component:
    """A sky component group: directions and apparent (beam-weighted) flux per channel."""

    name: str
    enu: np.ndarray  # [k, 3]
    flux: np.ndarray  # [k, nchan] Jy, beam applied


@dataclass
class SkySnapshot:
    t: Time
    chans: np.ndarray
    comps: list = field(default_factory=list)
    _tau: dict = field(default_factory=dict, repr=False)

    def matrix(self, ant, ci, names=None):
        """Model matrix for channel index ci over stands ant, optionally a subset of components."""
        f = self.chans[ci] * E.CHAN_BW
        M = np.zeros((len(ant), len(ant)), np.complex128)
        for c in self.comps:
            if names is not None and c.name not in names:
                continue
            if len(c.enu) == 0:
                continue
            # geometric delays (s) are cached per component and stand set; the
            # phases are evaluated in float32 (|f tau| < 1e3 turns, error < 1e-4 rad)
            key = (c.name, ant.tobytes())
            tau = self._tau.get(key)
            if tau is None:
                tau = ((stand_enu()[ant] @ c.enu.T) / SOL).astype(np.float64)
                self._tau[key] = tau
            ph = (2 * np.pi * f * tau).astype(np.float32)
            A = np.empty(ph.shape, np.complex64)
            np.cos(ph, out=A.real)
            np.sin(ph, out=A.imag)
            M += (A * c.flux[:, ci].astype(np.float32)) @ A.conj().T
        return M


def build_snapshot(t: Time, chans, pol="XX", nside=32, extra_sources=None, flux_scale=None,
                   offsets_deg=None, sun_flux_factor=1.0, beam_model="empirical", fluct_rms=0.0):
    """Sky snapshot for the given time.

    extra_sources: list of (name, enu, flux_at_74MHz, spectral_index) not in the model sky
    flux_scale: dict name -> multiplicative error applied to that point source
    offsets_deg: dict name -> (d_east, d_north) apparent shift in degrees (ionospheric refraction)
    """
    chans = np.asarray(chans)
    nu = chans * E.CHAN_BW
    nu_mhz = nu / 1e6
    snap = SkySnapshot(t, chans)
    # collect (name, enu[k,3], flux[k,nchan]) first; the beam is evaluated once
    # per frequency for all directions (lsl reloads its model on every call)
    parts = []
    names, enu, flux, free = sm.point_sources(t, nu_mhz)
    for n, e, s in zip(names, enu, flux):
        e = e.copy()
        if offsets_deg and n in offsets_deg:
            de, dn = np.radians(offsets_deg[n])
            e = e + np.array([de, dn, 0.0])
            e /= np.linalg.norm(e)
        if flux_scale and n in flux_scale:
            s = s * flux_scale[n]
        if n == "Sun":
            s = s * sun_flux_factor
        parts.append((n, e[None], s[None]))
    if extra_sources is not None:
        from astropy.coordinates import AltAz
        coords, s74, alpha = extra_sources
        aa = coords.transform_to(AltAz(obstime=t, location=sm.LWASV))
        es = sm.enu_from_altaz(aa.alt.rad, aa.az.rad)
        up = es[:, 2] > np.sin(np.radians(5))
        if up.any():
            parts.append(("extra", es[up], (s74[:, None] * (nu_mhz[None, :] / 74.0) ** alpha[:, None])[up]))
    if nside:
        denu, djy = sm.diffuse(t, nu_mhz, nside=nside, fluct_rms=fluct_rms)
        parts.append(("diffuse", denu, djy))
    if not parts:
        return snap
    allenu = np.concatenate([p[1] for p in parts])
    b = np.stack([dipole_beam(allenu, f, pol, beam_model) for f in nu], axis=1)  # [k, nchan]
    i0 = 0
    for n, e, fl in parts:
        snap.comps.append(Component(n, e, fl * b[i0:i0 + len(e)]))
        i0 += len(e)
    return snap


def random_extra_sources(n=25, seed=0):
    """Fixed-on-sky faint sources (100-800 Jy at 74 MHz) that the model lacks.
    Returns (SkyCoord[n], S74[n], alpha[n])."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    rng = np.random.default_rng(seed)
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-0.4, 1, n)))
    return SkyCoord(ra * u.deg, dec * u.deg), 10 ** rng.uniform(2, 2.9, n), rng.normal(-0.7, 0.15, n)
