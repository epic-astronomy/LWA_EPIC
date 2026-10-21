"""
Candidates at the positions of known sources.

Several sources in the sky of LWA-SV produce real, sometimes dispersed,
bursts that are not FRBs:

  solar, jovian   The Sun (type III bursts, which drift in frequency much
                  like a dispersed burst) and Jupiter (decametric emission,
                  below about 40 MHz). Their positions move, so they are
                  computed for each candidate with astropy's built-in
                  ephemeris (no download), cached for 60 s, and converted to
                  FK5 J2000, the frame of the image WCS.
  bright_source   The brightest radio sources (Cyg A, Cas A, Tau A, Vir A,
                  Her A, Hya A). Ionospheric scintillation modulates them
                  strongly at these frequencies, and a residual of the
                  normalisation can trigger the search.
  pulsar          Bright low-frequency pulsars. A single pulse is a genuine
                  dispersed burst; it is identified by its position and its
                  DM together. The Crab pulsar (B0531+21) is in the nebula
                  Tau A, so its giant pulses are told from the nebula's
                  scintillation by their DM (pulsars are checked first).

A match is made within a number of synthesised beams (at the band centre,
widened towards the horizon as the beam is, candidates.elevation_radius):
3 for the Sun and Jupiter, whose bursts can come from the corona or from
sidelobes around them, 1.5 for the bright sources and 2 for the pulsars. A
pulsar also needs |DM - DM_psr| <= max(3 DM steps, 10% of DM).

A candidate at a known source is stored with its data as class "known" and a
known_source record; it is not RFI, and the combiner opens no event for it.
"""
from __future__ import annotations

import csv
import threading

import numpy as np

from .candidates import elevation_radius, separation_deg
from .geometry import LWASV_LAT, LWASV_LON, altaz

BRIGHT_SOURCES = [  # name, RA, Dec (J2000, deg)
    ("Cyg A", 299.868, 40.734),
    ("Cas A", 350.850, 58.815),
    ("Tau A", 83.633, 22.014),
    ("Vir A", 187.706, 12.391),
    ("Her A", 252.784, 4.993),
    ("Hya A", 139.524, -12.096),
]

# Bright pulsars at low frequencies (ATNF catalogue): name, RA, Dec (J2000,
# sexagesimal), DM (pc cm^-3), period (s)
_PULSARS = [
    ("B0301+19", "03:04:33.1", "+19:32:51", 15.66, 1.38758),
    ("B0531+21", "05:34:31.97", "+22:00:52.1", 56.77, 0.0333924),
    ("B0320+39", "03:23:26.6", "+39:44:52", 26.19, 3.03207),
    ("B0329+54", "03:32:59.37", "+54:34:43.6", 26.764, 0.71452),
    ("B0809+74", "08:14:59.5", "+74:29:05.7", 5.733, 1.29224),
    ("B0823+26", "08:26:51.38", "+26:37:23.8", 19.454, 0.53066),
    ("B0834+06", "08:37:05.64", "+06:10:14.6", 12.889, 1.27377),
    ("B0943+10", "09:46:07.6", "+09:51:55", 15.32, 1.09771),
    ("B0950+08", "09:53:09.31", "+07:55:35.8", 2.970, 0.25307),
    ("B1133+16", "11:36:03.25", "+15:51:04.5", 4.84, 1.18791),
    ("B1237+25", "12:39:40.46", "+24:53:49.3", 9.25, 1.38245),
    ("B1508+55", "15:09:25.63", "+55:31:32.4", 19.62, 0.73968),
    ("B1919+21", "19:21:44.81", "+21:53:02.2", 12.44, 1.33730),
    ("B1929+10", "19:32:13.95", "+10:59:32.4", 3.18, 0.22652),
    ("B2016+28", "20:18:03.83", "+28:39:54.2", 14.17, 0.55795),
    ("B2217+47", "22:19:48.14", "+47:54:53.9", 43.52, 0.53847),
    ("B2310+42", "23:13:08.6", "+42:53:13", 17.28, 0.34943),
]


# common names, and the pulsars known for giant pulses at these frequencies
ALIASES = {"B0531+21": "Crab"}
GIANT_PULSES = {"B0531+21"}


def _sexagesimal(s: str) -> float:
    sign = -1.0 if s.strip().startswith("-") else 1.0
    parts = [abs(float(p)) for p in s.strip().lstrip("+-").split(":")]
    return sign * (parts[0] + parts[1] / 60.0 + (parts[2] if len(parts) > 2 else 0.0) / 3600.0)


def builtin_pulsars() -> list[dict]:
    """The built-in pulsar list as dicts (name, ra_deg, dec_deg, dm, period_s)"""
    return [dict(name=n, ra_deg=15.0 * _sexagesimal(ra), dec_deg=_sexagesimal(dec), dm=dm,
                 period_s=p, alias=ALIASES.get(n), giant_pulses=n in GIANT_PULSES)
            for n, ra, dec, dm, p in _PULSARS]


def load_pulsars(path: str | None = None) -> list[dict]:
    """
    The built-in pulsars, extended by a CSV with the header
    name,ra_deg,dec_deg,dm,period_s. A pulsar of the CSV replaces a built-in one
    of the same name.
    """
    psrs = {p["name"]: p for p in builtin_pulsars()}
    if path:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                name = (row.get("name") or "").strip()
                if not name or name.startswith("#"):
                    continue
                psrs[name] = dict(name=name, ra_deg=float(row["ra_deg"]),
                                  dec_deg=float(row["dec_deg"]), dm=float(row["dm"]),
                                  period_s=float(row["period_s"]) if (row.get("period_s") or
                                                                      "").strip() else None)
    return list(psrs.values())


class KnownSources:
    """
    beam_deg: the synthesised beam at the band centre. classify() returns the
    known source at a candidate's position (and DM), or None.
    """

    def __init__(self, beam_deg: float, pulsars_csv: str | None = None, body_beams: float = 3.0,
                 bright_beams: float = 1.5, pulsar_beams: float = 2.0,
                 body_cache_s: float = 60.0):
        self.beam = float(beam_deg)
        self.body_beams, self.bright_beams, self.pulsar_beams = body_beams, bright_beams, \
            pulsar_beams
        self.cache_s = body_cache_s
        self.pulsars = load_pulsars(pulsars_csv)
        self._psr_ra = np.array([p["ra_deg"] for p in self.pulsars])
        self._psr_dec = np.array([p["dec_deg"] for p in self.pulsars])
        self._psr_dm = np.array([p["dm"] for p in self.pulsars])
        self._src_ra = np.array([s[1] for s in BRIGHT_SOURCES])
        self._src_dec = np.array([s[2] for s in BRIGHT_SOURCES])
        self._bodies = None  # (unix, {name: (ra, dec)})
        self._lock = threading.Lock()

    def body_up(self, name: str, unix: float) -> bool:
        """
        Is the body above the horizon? A body below it cannot be what the
        monitor sees, however close its direction to a pixel low on the same
        side: the radius of the body test widens towards the horizon (four
        beams at most), which reaches tens of degrees from the set Sun.
        """
        ra, dec = self.body_positions(unix)[name]
        return float(altaz(ra, dec, unix)[0]) > 0.0

    def body_positions(self, unix: float) -> dict:
        """
        FK5 J2000 positions (deg) of the Sun and Jupiter as seen from LWA-SV,
        recomputed at most every body_cache_s (the Sun moves 2.5 arcmin in a
        minute, a small fraction of the smallest beam).
        """
        with self._lock:
            if self._bodies is not None and abs(unix - self._bodies[0]) <= self.cache_s:
                return self._bodies[1]
        import astropy.units as u
        from astropy.coordinates import (FK5, EarthLocation, UnitSphericalRepresentation,
                                         get_body, solar_system_ephemeris)
        from astropy.time import Time

        t = Time(unix, format="unix")
        loc = EarthLocation(lat=LWASV_LAT * u.deg, lon=LWASV_LON * u.deg, height=1477 * u.m)
        out = {}
        with solar_system_ephemeris.set("builtin"):
            for name, body in (("Sun", "sun"), ("Jupiter", "jupiter")):
                g = get_body(body, t, loc)  # topocentric GCRS, with distance
                # the apparent direction as that of a distant source, transformed as
                # the image WCS transforms directions. Keeping the distance would move
                # the origin to the barycentre, which for the Sun (0.005 AU from it)
                # gives a meaningless direction
                g = g.realize_frame(g.data.represent_as(UnitSphericalRepresentation))
                c = g.transform_to(FK5(equinox="J2000"))
                out[name] = (float(c.ra.deg), float(c.dec.deg))
        with self._lock:
            self._bodies = (unix, out)
        return out

    def classify(self, ra: float, dec: float, el: float, unix: float, dm: float | None = None,
                 dm_step: float = 0.0) -> dict | None:
        """
        The known source at (ra, dec), seen at elevation el (deg) at time unix,
        or None. A pulsar is checked first (it needs the DM as well), then the
        Sun and Jupiter, then the bright sources; within a kind the nearest wins.
        """
        scale = float(elevation_radius(self.beam, el))  # one beam, widened towards the horizon
        if dm is not None and len(self.pulsars):
            sep = separation_deg(ra, dec, self._psr_ra, self._psr_dec)
            tol = np.maximum(3.0 * dm_step, 0.1 * self._psr_dm)
            ok = (sep <= self.pulsar_beams * scale) & (np.abs(dm - self._psr_dm) <= tol)
            if ok.any():
                k = int(np.argmin(np.where(ok, sep, np.inf)))
                p = self.pulsars[k]
                return dict(kind="pulsar", name=p["name"], sep_deg=round(float(sep[k]), 3),
                            sep_beams=round(float(sep[k]) / scale, 2), dm_match=True,
                            dm_psr=p["dm"], period_s=p["period_s"], alias=p.get("alias"),
                            giant_pulse=bool(p.get("giant_pulses")))
        bodies = self.body_positions(unix)
        best = None
        for name, kind in (("Sun", "solar"), ("Jupiter", "jovian")):
            if not self.body_up(name, unix):
                continue
            bra, bdec = bodies[name]
            s = float(separation_deg(ra, dec, bra, bdec))
            if s <= self.body_beams * scale and (best is None or s < best["sep_deg"]):
                best = dict(kind=kind, name=name, sep_deg=round(s, 3),
                            sep_beams=round(s / scale, 2))
        if best is not None:
            return best
        sep = separation_deg(ra, dec, self._src_ra, self._src_dec)
        k = int(np.argmin(sep))
        if sep[k] <= self.bright_beams * scale:
            return dict(kind="bright_source", name=BRIGHT_SOURCES[k][0],
                        sep_deg=round(float(sep[k]), 3), sep_beams=round(float(sep[k]) / scale, 2))
        return None
