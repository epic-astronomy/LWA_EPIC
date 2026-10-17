"""Smooth multiplicative correction to the LSL dipole beam (fitted by research/beam_fit.py).

    B_true(az, el, f, pol) = B_LSL(az, el, f, pol) * C(az, el; p),   log C = sum_j p_j Y_j(az, el)

Basis, with rho = theta / 90 deg (theta the zenith angle) and phi the azimuth (from north
through east), all terms zero at the zenith (gauge: C = 1 at the zenith, so an overall
constant stays with the gains and the source flux scales):

    j  name    Y_j                       meaning
    0  r2      2 rho^2                   beam width (Zernike Z(2,0) - Z(2,0)(0))
    1  r4      6 rho^4 - 6 rho^2         mid-elevation shape, zero at zenith and horizon (Z(4,0) - 1)
    2  c1      rho cos(phi)              north-south tilt (Z(1,1))
    3  s1      rho sin(phi)              east-west tilt (Z(1,-1))
    4  c2      rho^2 cos(2 phi)          N-S versus E-W width (Z(2,2))
    5  s2      rho^2 sin(2 phi)          diagonal width (Z(2,-2))

The "radial" basis keeps r2 and r4 only.
"""
from __future__ import annotations

import numpy as np

NAMES = ("r2", "r4", "c1", "s1", "c2", "s2")
BASES = {"full": (0, 1, 2, 3, 4, 5), "radial": (0, 1), "r2c2": (0, 1, 4, 5)}


def basis(enu, terms=BASES["full"]):
    """Y_j at directions enu [P, 3] (east, north, up) -> [len(terms), P]."""
    enu = np.atleast_2d(enu)
    theta = np.arccos(np.clip(enu[:, 2], -1, 1))
    rho = theta / (np.pi / 2)
    phi = np.arctan2(enu[:, 0], enu[:, 1])
    allY = np.stack([2 * rho ** 2, 6 * rho ** 4 - 6 * rho ** 2, rho * np.cos(phi), rho * np.sin(phi),
                     rho ** 2 * np.cos(2 * phi), rho ** 2 * np.sin(2 * phi)])
    return allY[list(terms)]


def correction(enu, p, terms=BASES["full"]):
    """C(enu; p) [P]."""
    p = np.asarray(p, float)
    return np.exp(p @ basis(enu, terms))


def enu_from_azel(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], -1)


class Table:
    """Fitted coefficients per (ADP channel, pol) from beam_fit.py output; interpolated linearly in
    frequency between fitted channels (held constant outside). Callable as the beam_corr hook of
    forward.build_snapshot: corr(enu [P, 3], freq_hz, pol 'XX'|'YY') -> C [P]."""

    def __init__(self, path, half="sum"):
        res = np.load(path, allow_pickle=True).item()
        self.terms = tuple(res["meta"]["terms"])
        self.coef = {}  # pol -> (freq [n], p [n, J])
        for key, v in res.items():
            if key == "meta" or "fit" not in v:
                continue
            pol = int(key.split("_")[1])
            self.coef.setdefault(pol, []).append((v["freq"], v["fit"][half]["p"]))
        for pol in self.coef:
            rows = sorted(self.coef[pol], key=lambda x: x[0])
            self.coef[pol] = (np.array([x[0] for x in rows]), np.array([x[1] for x in rows]))

    def params(self, freq_hz, pol):
        f, P = self.coef[pol]
        return np.array([np.interp(freq_hz, f, P[:, j]) for j in range(P.shape[1])])

    def __call__(self, enu, freq_hz, pol):
        ip = 0 if pol in ("XX", 0) else 1
        return correction(enu, self.params(freq_hz, ip), self.terms)
