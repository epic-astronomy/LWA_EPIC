"""Sky model for the forward model: A-team point sources, the LFSM diffuse sky,
and the Sun as a component whose flux is solved for.

All flux densities are in Jy; diffuse brightness in Jy/sr. Directions are
returned as topocentric unit vectors (east, north, up) for a given time, so the
geometry code needs nothing from astropy at solve time.
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body
from astropy.time import Time

K_B = 1.380649e-23
C = 299792458.0

# Sky-model choices (environment, so the daemon and the research scripts share them):
#   STREAMCAL_ATEAM    dg2020_taubaars (default: de Gasperin et al. 2020 for Cas A, Cyg A and Vir A,
#                      Tau A as before) | dg2020 (de Gasperin for all four) | baars (Baars et al. 1977).
#                      On the TBF captures the fitted Cas A / Cyg A scale ratio is 0.96 with
#                      de Gasperin and 1.70 with Baars; Tau A fits ~1.5x the de Gasperin value.
#   STREAMCAL_CATALOG  "" (default) | vlssr: fixed compact sources from VLSSr (data/vlssr_bright.csv),
#                      part of the fixed (diffuse) model, not solved for
#   STREAMCAL_CATALOG_MIN   lower flux limit, Jy at 74 MHz (default 20)
#   STREAMCAL_CATALOG_ALPHA spectral index assumed below 74 MHz (default -0.75)
#   STREAMCAL_INPAINT_MIN   catalogue sources above this (Jy at 74 MHz, default 150) are also
#                           inpainted out of the LFSM map, like the A-team (r 3 deg, annulus 3-5 deg)
ATEAM_MODEL = os.environ.get("STREAMCAL_ATEAM", "dg2020_taubaars")
CATALOG = os.environ.get("STREAMCAL_CATALOG", "")
CATALOG_MIN = float(os.environ.get("STREAMCAL_CATALOG_MIN", "20"))
CATALOG_ALPHA = float(os.environ.get("STREAMCAL_CATALOG_ALPHA", "-0.75"))
INPAINT_MIN = float(os.environ.get("STREAMCAL_INPAINT_MIN", "150"))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# LWA-SV (lsl.common.stations.lwasv)
LWASV = EarthLocation.from_geodetic(lon=-106.885778 * u.deg, lat=34.348361 * u.deg, height=1477.8 * u.m)


def _baars(nu_mhz, a, b, c=0.0):
    x = np.log10(nu_mhz)
    return 10 ** (a + b * x + c * x * x)


def cyg_a(nu_mhz):
    # Baars et al. 1977, 20 MHz - 2 GHz
    return _baars(nu_mhz, 4.695, 0.085, -0.178)


def cas_a(nu_mhz, year=2026.8):
    # Baars et al. 1977 (epoch 1965.0) with the secular decrease
    # d = 0.97 - 0.30 log10(nu/GHz) per cent per year
    s1965 = _baars(nu_mhz, 5.625, -0.634, -0.023)
    rate = (0.97 - 0.30 * np.log10(nu_mhz / 1000.0)) / 100.0
    return s1965 * (1.0 - rate) ** (year - 1965.0)


def tau_a(nu_mhz):
    # Baars et al. 1977 above 1 GHz, extended to low frequency with the
    # observed flattening (about 2 kJy near 74 MHz, ~1.9-2.0 kJy in VLSS)
    return 1.6e3 * (nu_mhz / 74.0) ** -0.25


def vir_a(nu_mhz):
    # About 2.1 kJy at 74 MHz (VLSS), spectral index -0.8 below 300 MHz
    return 2.1e3 * (nu_mhz / 74.0) ** -0.8


def _read_csv(fn):
    """Small CSV with '#' comment lines and a header row -> dict of float arrays."""
    lines = [ln.strip() for ln in open(os.path.join(DATA, fn)) if ln.strip() and not ln.startswith("#")]
    cols = lines[0].split(",")
    vals = np.array([[float(x) for x in ln.split(",")] for ln in lines[1:]])
    return {c: vals[:, i] for i, c in enumerate(cols)}


@functools.lru_cache(maxsize=1)
def _dg2020_table():
    return _read_csv("ateam_degasperin2020.csv")


def _dg2020(name, nu_mhz):
    """de Gasperin et al. 2020 integrated spectrum, log-log interpolated over 30-77 MHz and
    extrapolated outside with the slope of the end 5 MHz. Cas A is faded from the model epoch
    (2015.65) at 0.7 per cent per year (an assumption: the Baars rate overestimates the
    low-frequency fading)."""
    t = _dg2020_table()
    ln, ls = np.log(t["nu_mhz"]), np.log(t[name])
    x = np.log(np.atleast_1d(nu_mhz).astype(float))
    y = np.interp(x, ln, ls)
    lo, hi = x < ln[0], x > ln[-1]
    s_lo = (ls[5] - ls[0]) / (ln[5] - ln[0])
    s_hi = (ls[-1] - ls[-6]) / (ln[-1] - ln[-6])
    y[lo] = ls[0] + s_lo * (x[lo] - ln[0])
    y[hi] = ls[-1] + s_hi * (x[hi] - ln[-1])
    out = np.exp(y)
    # below 30 MHz the de Gasperin data end; use the curvature of the Baars fit (which
    # includes the low-frequency flattening and Cas A's turnover towards ~20 MHz), scaled
    # to match at 30 MHz, instead of extrapolating the 30-35 MHz slope
    nu = np.exp(x)
    low = nu < 30.0
    if low.any() and name in _BAARS_SHAPE:
        b = _BAARS_SHAPE[name]
        out[low] = np.exp(np.interp(np.log(30.0), ln, ls)) * b(nu[low]) / b(30.0)
    if name == "CasA":
        out = out * (1 - 0.007) ** (_EPOCH_YEAR - 2015.65)
    return out if np.ndim(nu_mhz) else float(out[0])


_EPOCH_YEAR = 2026.8  # year at which Cas A is evaluated (overridden by set_epoch)
# spectral shapes used below 30 MHz (normalisation irrelevant; Cas A without secular fading)
_BAARS_SHAPE = {"CasA": lambda nu: _baars(nu, 5.625, -0.634, -0.023),
                "CygA": lambda nu: _baars(nu, 4.695, 0.085, -0.178),
                "VirA": lambda nu: (nu / 74.0) ** -0.8,
                "TauA": lambda nu: (nu / 74.0) ** -0.25}


def set_epoch(year):
    global _EPOCH_YEAR
    _EPOCH_YEAR = float(year)


def _ateam_spectrum(name, baars_fn):
    def f(nu_mhz):
        use_dg = ATEAM_MODEL == "dg2020" or (ATEAM_MODEL == "dg2020_taubaars" and name != "TauA")
        if use_dg:
            return _dg2020(name, nu_mhz)
        return baars_fn(nu_mhz, year=_EPOCH_YEAR) if name == "CasA" else baars_fn(nu_mhz)
    return f


@functools.lru_cache(maxsize=1)
def catalog_sources():
    """(SkyCoord[k], S74[k]) of the fixed catalogue sources, or None."""
    if CATALOG != "vlssr":
        return None
    t = _read_csv("vlssr_bright.csv")
    k = t["s74_jy"] >= CATALOG_MIN
    return SkyCoord(t["ra_deg"][k] * u.deg, t["dec_deg"][k] * u.deg), t["s74_jy"][k]


def catalog_fixed(t: Time, nu_mhz: np.ndarray, min_alt_deg=5.0):
    """Catalogue sources above min_alt: (enu[k, 3], flux[k, nchan]); empty arrays if none."""
    cat = catalog_sources()
    nu_mhz = np.atleast_1d(nu_mhz)
    if cat is None:
        return np.zeros((0, 3)), np.zeros((0, len(nu_mhz)))
    coords, s74 = cat
    aa = coords.transform_to(AltAz(obstime=t, location=LWASV))
    up = aa.alt.deg > min_alt_deg
    enu = enu_from_altaz(aa.alt.rad[up], aa.az.rad[up])
    return enu, s74[up, None] * (nu_mhz[None, :] / 74.0) ** CATALOG_ALPHA


@dataclass(frozen=True)
class PointSource:
    name: str
    coord: SkyCoord
    spectrum: object  # callable nu_mhz -> Jy
    free_flux: bool = False  # solve for its flux (Sun, flaring sources)


ATEAM = (
    PointSource("CygA", SkyCoord("19h59m28.36s", "+40d44m02.1s"), _ateam_spectrum("CygA", cyg_a)),
    PointSource("CasA", SkyCoord("23h23m24.0s", "+58d48m54s"), _ateam_spectrum("CasA", cas_a)),
    PointSource("TauA", SkyCoord("05h34m31.94s", "+22d00m52.2s"), _ateam_spectrum("TauA", tau_a)),
    PointSource("VirA", SkyCoord("12h30m49.42s", "+12d23m28.0s"), _ateam_spectrum("VirA", vir_a)),
)


def quiet_sun(nu_mhz):
    # Quiet Sun near solar minimum: ~1-2 sfu at 74 MHz, rising slowly with
    # frequency. Only a starting value: the Sun is solved as a free flux.
    return 1.5e4 * (nu_mhz / 74.0) ** 1.0


def enu_from_altaz(alt_rad, az_rad):
    ca = np.cos(alt_rad)
    return np.stack([ca * np.sin(az_rad), ca * np.cos(az_rad), np.sin(alt_rad)], axis=-1)


def point_sources(t: Time, nu_mhz: np.ndarray, min_alt_deg=5.0, include_sun=True):
    """Return (names, enu[k,3], flux[k,nchan], free[k]) for sources above min_alt."""
    frame = AltAz(obstime=t, location=LWASV)
    names, enus, flux, free = [], [], [], []
    srcs = list(ATEAM)
    if include_sun:
        srcs.append(PointSource("Sun", get_body("sun", t, LWASV), quiet_sun, free_flux=True))
    for s in srcs:
        aa = s.coord.transform_to(frame)
        if aa.alt.deg < min_alt_deg:
            continue
        names.append(s.name)
        enus.append(enu_from_altaz(aa.alt.rad, aa.az.rad))
        flux.append(s.spectrum(nu_mhz))
        free.append(s.free_flux)
    return names, np.array(enus).reshape(-1, 3), np.array(flux).reshape(-1, len(nu_mhz)), np.array(free, bool)


@functools.lru_cache(maxsize=4)
def _lfsm_native(nu_mhz_key: tuple):
    """LFSM at its native nside (256), K, Galactic RING, with the A-team inpainted."""
    from pygdsm import LowFrequencySkyModel

    lf = LowFrequencySkyModel(freq_unit="MHz")
    maps = np.atleast_2d(lf.generate(np.array(nu_mhz_key)))  # K, Galactic, RING
    return np.array([remove_compact(m) for m in maps])


@functools.lru_cache(maxsize=4)
def _lfsm_galactic(nside: int, nu_mhz_key: tuple):
    import healpy as hp

    return np.array([hp.ud_grade(m, nside) for m in _lfsm_native(nu_mhz_key)])


# Diffuse models selectable in diffuse(): "lfsm" (default; Dowell et al. 2017, a few
# degrees resolution) and "ovrolwa" (the OVRO-LWA m-mode maps of Eastwood et al. 2018,
# 15-26 arcmin, 36.5-73 MHz, hybrid with LFSM for the modes they lack; built by
# research/prep_ovrolwa.py), and "gsm16" (Zheng et al. 2017 via pygdsm, about 5 deg
# below 10 GHz; a control). The hybrid file is found through $STREAMCAL_OVROLWA_MAP.
DIFFUSE_MODELS = ("lfsm", "ovrolwa", "gsm16", "auto")
# "auto": OVRO-LWA inside AUTO_OVROLWA_MHZ (the maps span 36.5-73.2 MHz; the lower EPIC band
# at ~31 MHz and the top one at ~76 MHz are short extrapolations that still fit better than
# LFSM), LFSM outside (the 16 and 21 MHz bands) or when no hybrid file is installed.
AUTO_OVROLWA_MHZ = (30.0, 80.0)
# The hybrid file: $STREAMCAL_OVROLWA_MAP, else the first of these that exists (node-local
# disks; built by research/prep_ovrolwa.py). ovrolwa8 has all 8 release maps (36.5-73.2 MHz),
# ovrolwa_hybrid_n256 only the lower 4 (36.5-52.2 MHz).
OVROLWA_MAPS = ("/data/streamcal/diffuse_maps/ovrolwa8_hybrid_n256.npz",
                "/var/tmp/streamcal/diffuse_maps/ovrolwa8_hybrid_n256.npz",
                "/data/streamcal/diffuse_maps/ovrolwa_hybrid_n256.npz")
OVROLWA_MAP = OVROLWA_MAPS[0]


def ovrolwa_map_path():
    p = os.environ.get("STREAMCAL_OVROLWA_MAP")
    if p:
        return p
    for p in OVROLWA_MAPS:
        if os.path.exists(p):
            return p
    return None


def resolve_diffuse_model(model, nu_mhz):
    """'auto' -> 'ovrolwa' or 'lfsm' for these frequencies; other names pass through."""
    if model != "auto":
        return model
    nu = np.atleast_1d(nu_mhz)
    if ovrolwa_map_path() and AUTO_OVROLWA_MHZ[0] <= nu.min() and nu.max() <= AUTO_OVROLWA_MHZ[1]:
        return "ovrolwa"
    return "lfsm"


@functools.lru_cache(maxsize=1)
def _ovrolwa_nodes():
    import os

    p = ovrolwa_map_path()
    if p is None:
        raise FileNotFoundError("no OVRO-LWA hybrid map: set $STREAMCAL_OVROLWA_MAP or build it with "
                                "research/prep_ovrolwa.py into one of sky.OVROLWA_MAPS")
    z = np.load(p)
    return np.array(z["nu_mhz"], float), np.array(z["maps_k"], float)


@functools.lru_cache(maxsize=4)
def _ovrolwa_galactic(nside: int, nu_mhz_key: tuple):
    """OVRO-LWA hybrid map in K at the requested frequencies (Galactic, RING).

    Between the OVRO-LWA frequencies (36.528-52.224 MHz) each pixel is interpolated
    in log T - log nu. Outside that range (the lower EPIC band, 28-36.5 MHz) the
    nearest OVRO-LWA map is scaled per pixel by the LFSM spectrum,
    T(nu) = T_ovro(nu_0) LFSM(nu) / LFSM(nu_0): the small-scale structure is
    carried at the large-scale spectral index.
    """
    import healpy as hp

    nodes, maps = _ovrolwa_nodes()
    lt = np.log(np.maximum(maps, 1.0))
    ln = np.log(nodes)
    out = []
    for f in nu_mhz_key:
        if nodes[0] <= f <= nodes[-1]:
            k = int(np.clip(np.searchsorted(ln, np.log(f)) - 1, 0, len(ln) - 2))
            w = (np.log(f) - ln[k]) / (ln[k + 1] - ln[k])
            m = np.exp((1 - w) * lt[k] + w * lt[k + 1])
        else:
            k = 0 if f < nodes[0] else len(nodes) - 1
            lf = _lfsm_native((float(nodes[k]), float(f)))
            m = maps[k] * np.maximum(lf[1], 1.0) / np.maximum(lf[0], 1.0)
        out.append(hp.ud_grade(_remove_catalog_fine(m), nside))
    return np.array(out)


def _remove_catalog_fine(m, r_in_deg=0.75, r_out_deg=1.25):
    """Inpaint the fixed catalogue sources (STREAMCAL_CATALOG) out of a 15-26 arcmin map.

    The OVRO-LWA release removed only the eight brightest sources; the rest are in the map
    and would be counted twice if they are also model point sources."""
    import healpy as hp

    cat = catalog_sources()
    if cat is None:
        return m
    m = np.array(m, float)
    nside = hp.npix2nside(len(m))
    for c in cat[0]:
        g = c.galactic
        v = hp.ang2vec(g.l.deg, g.b.deg, lonlat=True)
        inner = hp.query_disc(nside, v, np.radians(r_in_deg))
        ring = np.setdiff1d(hp.query_disc(nside, v, np.radians(r_out_deg)), inner)
        m[inner] = np.median(m[ring])
    return m


@functools.lru_cache(maxsize=4)
def _gsm16_galactic(nside: int, nu_mhz_key: tuple):
    import healpy as hp
    from pygdsm import GlobalSkyModel16

    gsm = GlobalSkyModel16(freq_unit="MHz", data_unit="TRJ", resolution="low")
    maps = np.atleast_2d(gsm.generate(np.array(nu_mhz_key)))  # K (Rayleigh-Jeans), Galactic, RING
    return np.array([hp.ud_grade(remove_compact(m), nside) for m in maps])


def galactic_map(model: str, nside: int, nu_mhz_key: tuple):
    """Diffuse brightness temperature [nodes, npix] in K, Galactic RING, A-team removed."""
    if model == "lfsm":
        return _lfsm_galactic(nside, nu_mhz_key)
    if model == "ovrolwa":
        return _ovrolwa_galactic(nside, nu_mhz_key)
    if model == "gsm16":
        return _gsm16_galactic(nside, nu_mhz_key)
    raise ValueError(f"unknown diffuse model {model!r}; choose from {DIFFUSE_MODELS}")


def remove_compact(m, r_in_deg=6.0, r_out_deg=9.0):
    """Inpaint the A-team out of a Galactic HEALPix map.

    The LFSM contains Cyg A, Cas A, Tau A and Vir A smoothed to the map resolution
    (about 12 kJy above the surroundings within 2 deg of Cyg A and of Cas A at 38 MHz).
    They are modelled separately as point sources, so leaving them in the map counts
    them twice: on LWA-SV TBF data this drove the fitted Cyg A flux to ~0.3 of the
    catalogue. Pixels within r_in of each source take the median of the r_in-r_out annulus.
    """
    import healpy as hp

    m = np.array(m, float)
    nside = hp.npix2nside(len(m))
    discs = [(s.coord.galactic, r_in_deg, r_out_deg) for s in ATEAM]
    cat = catalog_sources()
    if cat is not None:  # bright catalogue sources are in the map too
        coords, s74 = cat
        discs += [(c.galactic, 3.0, 5.0) for c in coords[s74 >= INPAINT_MIN]]
    for g, rin, rout in discs:
        v = hp.ang2vec(g.l.deg, g.b.deg, lonlat=True)
        inner = hp.query_disc(nside, v, np.radians(rin))
        ring = np.setdiff1d(hp.query_disc(nside, v, np.radians(rout)), inner)
        m[inner] = np.median(m[ring])
    return m


@functools.lru_cache(maxsize=4)
def _fluct(nside: int, rms: float, seed: int):
    """Multiplicative small-scale structure (l ~ 30-400, ~1-6 deg) absent from the LFSM."""
    import healpy as hp

    ell = np.arange(3 * nside)
    cl = np.where(ell >= 30, (np.maximum(ell, 1) / 30.0) ** -2.5, 0.0)
    np.random.seed(seed)
    m = hp.synfast(cl, nside)
    return 1.0 + rms * m / m.std()


def diffuse(t: Time, nu_mhz: np.ndarray, nside=64, min_alt_deg=0.0, nu_nodes=None, fluct_rms=0.0, fluct_seed=1,
            model="auto"):
    """Diffuse sky above the horizon (model: one of DIFFUSE_MODELS; "auto" by default).

    Returns (enu[p,3], jy_per_pix[p,nchan]). The map is evaluated at a few
    frequency nodes and interpolated in log-log, since one EPIC band is only
    ~3 MHz wide.
    """
    import healpy as hp

    nu_mhz = np.atleast_1d(nu_mhz)
    model = resolve_diffuse_model(model, nu_mhz)
    if nu_nodes is None:
        nu_nodes = np.unique(np.round([nu_mhz.min(), nu_mhz.mean(), nu_mhz.max()], 3))
    tk = galactic_map(model, nside, tuple(float(x) for x in nu_nodes))  # [nodes, npix] K
    if fluct_rms > 0:
        tk = tk * _fluct(nside, fluct_rms, fluct_seed)[None, :]
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    gal = SkyCoord(l=phi * u.rad, b=(np.pi / 2 - theta) * u.rad, frame="galactic")
    aa = gal.transform_to(AltAz(obstime=t, location=LWASV))
    up = aa.alt.deg > min_alt_deg
    enu = enu_from_altaz(aa.alt.rad[up], aa.az.rad[up])
    omega = hp.nside2pixarea(nside)
    if len(nu_nodes) == 1:
        tk_nu = np.repeat(tk[:, up].T, len(nu_mhz), axis=1)
    else:
        lt = np.log(np.maximum(tk[:, up], 1.0))
        ln = np.log(nu_nodes)
        tk_nu = np.empty((up.sum(), len(nu_mhz)))
        for j, f in enumerate(nu_mhz):
            k = np.clip(np.searchsorted(ln, np.log(f)) - 1, 0, len(ln) - 2)
            w = (np.log(f) - ln[k]) / (ln[k + 1] - ln[k])
            tk_nu[:, j] = np.exp((1 - w) * lt[k] + w * lt[k + 1])
    lam = C / (nu_mhz * 1e6)
    jy = 2 * K_B * tk_nu / lam[None, :] ** 2 * omega / 1e-26
    return enu, jy
