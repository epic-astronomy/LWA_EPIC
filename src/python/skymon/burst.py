"""
Model of a dispersed burst as it appears in a sequence of EPIC images.

Used by the sky simulator and by the monitor's live injector, so a test
injection and a simulated sky go through exactly the same code.

In channel c (edges f1 < f2, MHz) a burst arriving at t_inf at infinite
frequency is received over [t_inf + K DM / f2^2, t_inf + K DM / f1^2 + width]:
the sweep across the channel plus the intrinsic width. Its fluence is spread
uniformly over that interval, and an image covering [t, t + dt) receives the
overlapping fraction. On the sky it is a point source, which the image shows
with a Gaussian beam of FWHM lambda / D.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .dedisp import K_DM
from .geometry import beam_fwhm_deg


@dataclass
class Burst:
    id: str
    ra: float  # deg J2000
    dec: float
    dm: float  # pc cm^-3
    t_inf: float  # unix arrival time at infinite frequency
    width_s: float = 0.005  # intrinsic width
    snr: float = 15.0  # target S/N used to set the amplitude
    amp_sigma: float = 0.0  # fluence per channel, in units of the per-image noise
    kind: str = "injection"  # injection (live test) or simulated (sky simulator truth)
    meta: dict = field(default_factory=dict)

    def arrival(self, f_mhz):
        return self.t_inf + K_DM * self.dm * np.asarray(f_mhz, float) ** -2

    def span(self, edges_mhz):
        return float(self.arrival(edges_mhz[-1])), float(self.arrival(edges_mhz[0]) + self.width_s)

    def to_dict(self):
        return asdict(self)


def channel_fractions(burst: Burst, edges_mhz: np.ndarray, t0: float, dt: float) -> np.ndarray:
    """Fraction of each channel's fluence that falls in the image [t0, t0 + dt)"""
    a = burst.arrival(edges_mhz[1:])  # top edge of each channel arrives first
    b = burst.arrival(edges_mhz[:-1]) + burst.width_s
    ov = np.clip(np.minimum(b, t0 + dt) - np.maximum(a, t0), 0.0, None)
    return ov / np.maximum(b - a, 1e-9)


def effective_samples(burst: Burst, edges_mhz: np.ndarray, dt: float) -> float:
    """Samples a channel's signal occupies at resolution dt, including straddling"""
    smear = float(np.mean(burst.arrival(edges_mhz[:-1]) - burst.arrival(edges_mhz[1:])))
    return (smear + burst.width_s) / dt + 1.0


def amplitude_for_snr(snr: float, nchan: int, n_eff: float) -> float:
    """
    Fluence per channel, in units of the per-sample noise, that gives a dedispersed
    boxcar S/N of snr when every channel's signal occupies n_eff samples:
    S/N = a C / sqrt(C n_eff)
    """
    return snr * math.sqrt(n_eff / nchan)


def stamp(image: np.ndarray, burst: Burst, x: float, y: float, amp: np.ndarray,
          frac: np.ndarray, centres_mhz: np.ndarray, pix_deg: float, radius: float = 3.0):
    """
    Add the burst to image [C][y][x] (FITS order) at 1-based pixel (x, y).
    amp [C] is the channel fluence in image units summed over all images; frac [C]
    the fraction in this image. The beam peak is 1 so the sky pixel at the
    position receives amp * frac.
    """
    act = np.nonzero(frac > 0)[0]
    if len(act) == 0:
        return
    G = image.shape[-1]
    sig = beam_fwhm_deg(float(np.max(centres_mhz[act]))) / pix_deg / 2.3548
    r = int(math.ceil(radius * beam_fwhm_deg(float(np.min(centres_mhz[act]))) / pix_deg))
    x0, y0 = x - 1.0, y - 1.0
    xs = np.arange(max(0, int(x0) - r), min(G, int(x0) + r + 2))
    ys = np.arange(max(0, int(y0) - r), min(G, int(y0) + r + 2))
    if len(xs) == 0 or len(ys) == 0:
        return
    dx2 = (xs[None, :] - x0) ** 2
    dy2 = (ys[:, None] - y0) ** 2
    for c in act:
        s = sig * centres_mhz.max() / centres_mhz[c]  # beam scales with wavelength
        psf = np.exp(-(dx2 + dy2) / (2 * s * s))
        image[c, ys[0]: ys[-1] + 1, xs[0]: xs[-1] + 1] += (amp[c] * frac[c]) * psf
