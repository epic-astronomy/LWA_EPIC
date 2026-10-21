"""The calibration record for the all-sky monitor (docs/skymon/streamcal-interface.md).

One JSON record per published gain table, keyed by its gain_version, written
atomically beside the gain table (--calinfo). It carries what the monitor
cannot measure itself to turn a candidate's signal into a fluence:

- C1 gain_version of the table, and the time it was solved for;
- C2 the absolute scale F_X, F_Y per channel that the relative table removed
  (data counts per Jy^1/2: the median gain amplitude the normalisation took
  out, StreamCal.flux_scale);
- C3 the calibrators of the solution: name, J2000 position, the intrinsic
  model spectrum on the solved channels, the flux scale solved for it and
  whether its flux is free (the Sun);
- C5 the quality of the solution: chi^2/dof, the cut, channels rejected as
  interference, stands flagged.

No beam table is written (C6): the monitor then uses its own dipole model.
"""
from __future__ import annotations

import json
import os

import numpy as np
from astropy.coordinates import GCRS, SkyCoord, get_body
from astropy.time import Time

from . import sky as sm


def _source_coords(names, t: Time):
    """J2000 (ICRS) position of each named source at time t"""
    out = {}
    for s in sm.ATEAM:
        if s.name in names:
            c = s.coord.icrs
            out[s.name] = (float(c.ra.deg), float(c.dec.deg))
    if "Sun" in names:
        # the apparent direction from the station, as that of a distant source; .icrs of
        # the body itself would be its direction from the solar system's barycentre
        g = get_body("sun", t, sm.LWASV)
        c = SkyCoord(g.ra, g.dec, frame=GCRS(obstime=t)).icrs
        out["Sun"] = (float(c.ra.deg), float(c.dec.deg))
    return out


def record(sc, version: int, t_unix: float, chi=None, chi_cut=None) -> dict:
    """The record of the gain table just published by StreamCal sc as version"""
    t = Time(t_unix, format="unix")
    nu_mhz = np.asarray(sc.freq, float) / 1e6  # solved channels
    names, enu, flux, free = sm.point_sources(t, nu_mhz)
    sky = getattr(sc, "last_sky", None) or {}
    scale = dict(zip(sky.get("names", []), np.atleast_1d(sky.get("scale", []))))
    offset = dict(zip(sky.get("names", []), np.atleast_2d(sky.get("offset_deg", np.zeros((0, 2))))))
    pos = _source_coords(set(names), t)
    cals = []
    for n, e, f, fr in zip(names, enu, flux, free):
        if n not in pos:
            continue
        alt = float(np.degrees(np.arcsin(np.clip(e[2], -1, 1))))
        az = float(np.degrees(np.arctan2(e[0], e[1])) % 360)
        s = scale.get(n)
        cals.append(dict(name=n, ra_deg=pos[n][0], dec_deg=pos[n][1], alt_deg=round(alt, 3), az_deg=round(az, 3),
                         model_jy=[float(x) for x in f], free_flux=bool(fr),
                         scale=None if s is None or not np.isfinite(s) else float(s),
                         offset_deg=[float(x) for x in offset.get(n, (0.0, 0.0))]))
    fs = getattr(sc, "flux_scale", None)
    rec = dict(layout=1, gain_version=int(version), applied=True, t_solved_unix=float(t_unix),
               chan0=int(sc.chan0), nchan=int(sc.nchan),
               solve_chans=[int(sc.chan0 + i) for i in sc.solve_idx],
               flux_scale=dict(X=[float(x) for x in fs[0]], Y=[float(x) for x in fs[1]],
                               units="data counts per Jy^1/2"),
               calibrators=cals, model="streamcal forward model", solver="streamcal stefcal+kalman")
    if chi is not None:
        chi = np.asarray(chi, float)
        cut = float(chi_cut) if chi_cut is not None else float("nan")
        bad = ~np.isfinite(chi) | (chi > cut)
        flag = getattr(sc, "flag_state", None)
        rec["quality"] = dict(
            chi2_median=float(np.nanmedian(chi)) if np.isfinite(chi).any() else None, chi2_cut=cut,
            rfi_channels=[int(sc.chan0 + sc.solve_idx[k]) for k in np.nonzero(bad.any(axis=0))[0]],
            flagged_stands=dict(X=int(flag[0].sum()) if flag is not None else None,
                                Y=int(flag[1].sum()) if flag is not None else None))
    return rec


def write(path: str, rec: dict) -> None:
    """Replace the record at path atomically (the monitor reads it whenever it changes)"""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, path)
