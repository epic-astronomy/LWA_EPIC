"""
Simulated EPIC sky for testing the all-sky monitor end to end.

Produces Stokes I images in the imager's format and order (FITS [chan][y][x],
same WCS, same time stamps and image length) with

* a diffuse sky brighter towards the Galactic plane, falling as nu^-2.55,
* Cyg A, Cas A, Tau A and Vir A as point sources in a beam of lambda / D,
* a primary beam proportional to sin(el)^1.6 and nothing below the horizon,
* radiometer noise per pixel and channel proportional to the sky brightness,
* optional RFI: whole-sky broadband flashes and a noisy channel,
* FRBs at random positions, times, DMs, widths and S/N (truth in a JSONL file).

The images can be written to a shared-memory ring at the real image rate,
where a monitor reads them exactly as it reads the imager, or handed to a
monitor in the same process as fast as it can take them.
"""
from __future__ import annotations

import json
import math
import os
import time
import uuid

import numpy as np

from .burst import Burst, amplitude_for_snr, channel_fractions, effective_samples, stamp
from .dedisp import K_DM
from .geometry import (FS, SEQ_MULT_S, Band, altaz, beam_fwhm_deg, image_wcs, seq_to_unix,
                       time_to_set)
from .shm import ImageMeta

SOURCES = [  # name, RA, Dec (J2000, deg), flux relative to the diffuse sky at 60 MHz
    ("Cyg A", 299.868, 40.734, 6.0),
    ("Cas A", 350.850, 58.815, 6.0),
    ("Tau A", 83.633, 22.014, 1.2),
    ("Vir A", 187.706, 12.391, 1.5),
]


class SkySimulator:
    def __init__(self, chan0=2400 - 64, nchan=128, grid=128, grid_res=1.056, nseqs=2000,
                 start_unix=None, phase_mode="plane", noise_factor=3.0, frb_per_hour=30.0,
                 snr=(8.0, 30.0), dm=(20.0, 3000.0), width_s=(0.001, 0.05), el_min=15.0,
                 rfi_per_hour=6.0, rfi_channel=True, truth_path=None, seed=None,
                 tier_dt=None, model_interval_s=10.0, fft_workers=4, frb_list=None):
        self.band = Band(chan0, nchan, 1)
        self.grid, self.grid_res, self.nseqs = grid, grid_res, nseqs
        self.dt = nseqs * SEQ_MULT_S / FS
        start_unix = time.time() if start_unix is None else start_unix
        self.seq0 = int(round(start_unix * FS / SEQ_MULT_S / nseqs)) * nseqs
        self.phase_mode = phase_mode
        self.noise_factor = noise_factor
        self.frb_rate = frb_per_hour / 3600.0
        self.snr, self.dm, self.width = snr, dm, width_s
        self.el_min = el_min
        self.rfi_rate = rfi_per_hour / 3600.0
        self.rfi_channel = rfi_channel
        self.rng = np.random.default_rng(seed)
        self.truth_path = truth_path
        self.model_interval = model_interval_s
        self.fft_workers = fft_workers
        # FRBs from a list shared by the simulators of several bands (make_frb_list)
        # instead of drawn at random: the same bursts reach every band
        # in order of arrival at the top of this band, which is not the order of
        # t_inf: a burst of high DM reaches a low band long after one of low DM
        f_top = float(self.band.edges_mhz()[-1])
        self.frb_list = sorted(frb_list or [], key=lambda f: f["t_inf"] + K_DM * f["dm"] / f_top ** 2)
        if self.frb_list:
            self.frb_rate = 0.0
        self.tier_dt = tier_dt  # function dm -> (tier dt, decim) used to set amplitudes
        self.n = 0
        self.model_t = None
        self.active: list[Burst] = []
        self.truth: list[dict] = []
        self.edges = self.band.edges_mhz()
        self.centres = self.band.centres_mhz()
        sweep = K_DM * (self.edges[0] ** -2 - self.edges[-1] ** -2)
        self.sweep = sweep
        self.next_frb = self._t(0) + self.rng.exponential(1 / self.frb_rate) \
            if self.frb_rate > 0 else math.inf
        self.next_rfi = self._t(0) + self.rng.exponential(1 / self.rfi_rate) \
            if self.rfi_rate > 0 else math.inf
        if truth_path:
            os.makedirs(os.path.dirname(os.path.abspath(truth_path)), exist_ok=True)
            open(truth_path, "w").close()

    def _t(self, n):
        return seq_to_unix(self.seq0 + n * self.nseqs)

    # ------------------------------------------------------------ sky model
    def _model(self, unix):
        """
        Sky snapshots at unix and unix + model_interval; images in between
        interpolate linearly, so the sky turns smoothly as in real images.
        """
        if self.model_t is not None and unix < self._snap_b["t"] + self.model_interval:
            a = self._snap_b  # chain the snapshots, one new one per interval
        else:
            a = self._snapshot(unix)
        self._snap_a, self._snap_b = a, self._snapshot(a["t"] + self.model_interval)
        self.model_t = a["t"]
        self.wcs = a["wcs"]
        self.ra_grid, self.dec_grid, self.el_grid = a["ra"], a["dec"], a["el"]

    def _interp(self, t0):
        a, b = self._snap_a, self._snap_b
        f = np.float32((t0 - a["t"]) / (b["t"] - a["t"]))
        self.model = a["model"] + f * (b["model"] - a["model"])
        self.sigma = a["sigma"] + f * (b["sigma"] - a["sigma"])

    def _snapshot(self, unix):
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        G = self.grid
        wcs = image_wcs(unix, G, self.grid_res, self.phase_mode)
        yy, xx = np.mgrid[1:G + 1, 1:G + 1]
        ra, dec = wcs.all_pix2world(xx.ravel(), yy.ravel(), 1, ra_dec_order=True)
        good = np.isfinite(ra) & np.isfinite(dec)
        ra = np.where(good, ra, 0.0)
        dec = np.where(good, dec, 0.0)
        el, _ = altaz(ra, dec, unix)
        el = np.where(good, el, -90.0)
        gal = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="fk5").galactic
        b = gal.b.deg
        lcen = np.radians(gal.l.deg)
        diffuse = 1.0 + 3.0 * np.exp(-0.5 * (b / 8.0) ** 2) * (0.6 + 0.4 * np.cos(lcen)) + \
            2.0 * np.exp(-0.5 * ((np.degrees(np.arccos(np.cos(np.radians(b)) * np.cos(lcen))))
                                 / 25.0) ** 2)
        pb = np.where(el > 0, np.sin(np.radians(np.clip(el, 0, 90))) ** 1.6, 0.0)
        spec = (self.centres / 60.0) ** -2.55
        model = spec[:, None] * (diffuse * pb)[None, :]
        # point sources, Gaussian beam per channel
        pix = self.grid_res
        for _, sra, sdec, flux in SOURCES:
            x, y = wcs.all_world2pix(sra, sdec, 1, ra_dec_order=True)
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            sel, _ = altaz(sra, sdec, unix)
            if sel <= 0:
                continue
            g = np.sin(np.radians(sel)) ** 1.6
            r2 = (xx.ravel() - x) ** 2 + (yy.ravel() - y) ** 2
            for c, f in enumerate(self.centres):
                s = beam_fwhm_deg(f) / pix / 2.3548
                model[c] += flux * g * spec[c] * np.exp(-r2 / (2 * s * s))
        model = model.reshape(len(self.centres), G, G).astype(np.float32)
        # radiometer noise per image: T / sqrt(B tau), scaled for imaging sidelobes
        nbt = math.sqrt(self.band.chan_bw_hz * self.dt)
        floor = 0.02 * spec[:, None, None]
        horizon = (el.reshape(G, G) > 0)[None]
        sigma = ((model + floor) * self.noise_factor / nbt * horizon).astype(np.float32)
        return dict(t=unix, wcs=wcs, model=model, sigma=sigma, ra=ra, dec=dec, el=el)

    def _noise(self, C, G):
        """Unit-variance noise correlated on the scale of the beam, as in an image"""
        if getattr(self, "_nk", None) is None:
            s = beam_fwhm_deg(float(np.mean(self.centres))) / self.grid_res / 2.3548
            k = np.fft.fftfreq(G)
            kr = np.fft.rfftfreq(G)
            g = np.exp(-2 * (np.pi * s) ** 2 * (k[:, None] ** 2 + kr[None, :] ** 2))
            # normalise so the correlated noise has unit variance per pixel
            full = np.exp(-2 * (np.pi * s) ** 2 * (k[:, None] ** 2 + k[None, :] ** 2))
            self._nk = (g / np.sqrt(np.mean(full ** 2))).astype(np.complex64)
        import scipy.fft as sfft

        w = self.rng.standard_normal((C, G, G), dtype=np.float32)
        f = sfft.rfft2(w, workers=self.fft_workers)
        f *= self._nk
        return sfft.irfft2(f, s=(G, G), workers=self.fft_workers)

    # ------------------------------------------------------------ events
    def _new_frb(self, unix):
        import healpy as hp
        G = self.grid
        for _ in range(50):
            k = int(self.rng.integers(G * G))
            if self.el_grid[k] < self.el_min:
                continue
            ra = float(self.ra_grid[k] + self.rng.uniform(-0.5, 0.5) * self.grid_res)
            dec = float(np.clip(self.dec_grid[k] + self.rng.uniform(-0.5, 0.5) * self.grid_res,
                                -89, 89))
            left = float(time_to_set(dec, ra % 360, unix, self.el_min - 2))
            dm_hi = min(self.dm[1], (left - 120.0) / self.sweep)
            if dm_hi > self.dm[0]:
                break
        else:
            return None
        ra %= 360.0
        dm = float(math.exp(self.rng.uniform(math.log(self.dm[0]), math.log(dm_hi))))
        t_top = unix + 2.0
        t_inf = t_top - K_DM * dm / self.edges[-1] ** 2
        width = float(math.exp(self.rng.uniform(math.log(self.width[0]), math.log(self.width[1]))))
        snr = float(self.rng.uniform(*self.snr))
        b = Burst(id="sim-" + uuid.uuid4().hex[:10], ra=ra, dec=dec, dm=dm, t_inf=t_inf,
                  width_s=width, snr=snr, kind="simulated")
        tdt, decim = self.tier_dt(dm) if self.tier_dt else (self.dt, 1)
        n_eff = effective_samples(b, self.edges, tdt)
        b.amp_sigma = amplitude_for_snr(snr, len(self.centres), n_eff)
        x, y = self.wcs.all_world2pix(ra, dec, 1, ra_dec_order=True)
        ix = int(np.clip(round(float(x)) - 1, 0, G - 1))
        iy = int(np.clip(round(float(y)) - 1, 0, G - 1))
        sig = self.sigma[:, iy, ix].astype(np.float64)
        b._amp = b.amp_sigma * sig * decim ** 0.5
        span = b.span(self.edges)
        b.meta = dict(span=list(span), tier_dt=tdt, n_eff=n_eff, el=float(altaz(ra, dec, unix)[0]))
        self.active.append(b)
        rec = b.to_dict()
        rec["t_bottom"] = float(b.arrival(self.edges[0]))
        self.truth.append(rec)
        if self.truth_path:
            with open(self.truth_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return b

    def _listed_frb(self, spec, unix):
        """
        Start a burst from the shared list; its amplitude comes from this band's
        noise. With a spectrum in the list, the S/N in this band is the listed
        one times the spectrum at the band centre; scattering then spreads the
        same fluence over tau(nu) more, which lowers the S/N found.
        """
        nu = float(np.mean(self.centres))
        factor = spectral_factor(spec.get("spectrum"), nu)
        snr = float(spec["snr"]) * factor
        if snr < 0.2:
            return None  # no emission in this band
        tau = float(spec.get("tau_1ghz", 0.0)) * (nu / 1000.0) ** -4
        b = Burst(id=spec["id"], ra=float(spec["ra"]), dec=float(spec["dec"]), dm=float(spec["dm"]),
                  t_inf=float(spec["t_inf"]), width_s=float(spec.get("width_s", 0.005)),
                  snr=snr, kind="simulated")
        x, y = self.wcs.all_world2pix(b.ra, b.dec, 1, ra_dec_order=True)
        G = self.grid
        if not (np.isfinite(x) and np.isfinite(y)) or altaz(b.ra, b.dec, unix)[0] <= 0:
            return None
        tdt, decim = self.tier_dt(b.dm) if self.tier_dt else (self.dt, 1)
        n_eff = effective_samples(b, self.edges, tdt)
        b.amp_sigma = amplitude_for_snr(b.snr, len(self.centres), n_eff)
        ix = int(np.clip(round(float(x)) - 1, 0, G - 1))
        iy = int(np.clip(round(float(y)) - 1, 0, G - 1))
        b._amp = b.amp_sigma * self.sigma[:, iy, ix].astype(np.float64) * decim ** 0.5
        # scattering: the same fluence over a longer time (after the amplitude is set)
        b.width_s += tau
        n_scat = effective_samples(b, self.edges, tdt)
        b.meta = dict(span=list(b.span(self.edges)), tier_dt=tdt, n_eff=n_eff,
                      el=float(altaz(b.ra, b.dec, unix)[0]), spectral_factor=factor,
                      tau_s=tau, snr_expected=snr * math.sqrt(n_eff / n_scat))
        self.active.append(b)
        rec = b.to_dict()
        rec["t_bottom"] = float(b.arrival(self.edges[0]))
        self.truth.append(rec)
        if self.truth_path:
            with open(self.truth_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return b

    # ------------------------------------------------------------ images
    def next_image(self):
        n = self.n
        t0 = self._t(n)
        if self.model_t is None or t0 >= self._snap_b["t"]:
            self._model(t0)
        self._interp(t0)
        C, G = len(self.centres), self.grid
        img = self._noise(C, G)
        img *= self.sigma
        img += self.model
        if self.rfi_channel:
            c = C // 3
            img[c] += self.rng.standard_normal((G, G), dtype=np.float32) * 4 * self.sigma[c]
        if t0 >= self.next_rfi:
            self.next_rfi = t0 + self.rng.exponential(1 / self.rfi_rate)
            amp = self.rng.uniform(5, 50)
            chans = slice(None) if self.rng.random() < 0.5 else slice(
                int(self.rng.integers(C - 16)), None)
            img[chans] += amp * self.sigma[chans]
        while t0 >= self.next_frb:
            self.next_frb += self.rng.exponential(1 / self.frb_rate)
            self._new_frb(t0)
        while self.frb_list and \
                self.frb_list[0]["t_inf"] + K_DM * self.frb_list[0]["dm"] / self.edges[-1] ** 2 \
                < t0 + self.dt + 1.0:
            self._listed_frb(self.frb_list.pop(0), t0)
        keep = []
        for b in self.active:
            span = b.meta["span"]
            if t0 > span[1]:
                continue
            keep.append(b)
            if t0 + self.dt < span[0]:
                continue
            frac = channel_fractions(b, self.edges, t0, self.dt)
            if frac.any():
                x, y = self.wcs.all_world2pix(b.ra, b.dec, 1, ra_dec_order=True)
                stamp(img, b, float(x), float(y), b._amp, frac, self.centres, self.grid_res)
        self.active = keep
        meta = ImageMeta(n, self.seq0 + n * self.nseqs, self.nseqs, self.band.chan0,
                         self.band.nchan, self.band.nbin, G, self.phase_mode, self.grid_res,
                         self.dt * 1e3)
        self.n += 1
        return meta, img


def tier_dt_function(edges_mhz, dt, nchan, dm_max=3000.0, block0=256):
    """dm -> (tier dt, decimation) for the monitor's default tiers, for setting S/N"""
    from .dedisp import make_tiers

    tiers = make_tiers(float(edges_mhz[0]), float(edges_mhz[-1]), dt, nchan, dm_max, None,
                       block0)

    def f(dm):
        for t in tiers:
            if dm <= t.dm_max:
                return t.dt, t.decim
        return tiers[-1].dt, tiers[-1].decim
    return f


def run_shm(sim: SkySimulator, ring_name: str, duration_s: float, nslots: int = 64,
            realtime: bool = True, log_every_s: float = 60.0):
    """Write images to a shared-memory ring at the real image rate"""
    from .shm import RingWriter

    w = RingWriter(ring_name, nslots, len(sim.centres), sim.grid)
    n_total = int(duration_s / sim.dt)
    wall0 = time.time()
    late = 0.0
    try:
        for i in range(n_total):
            meta, img = sim.next_image()
            if realtime:
                target = wall0 + (i + 1) * sim.dt
                d = target - time.time()
                if d > 0:
                    time.sleep(d)
                else:
                    late = max(late, -d)
            w.write(img, meta.seq_start, meta.nseqs, meta.chan0, meta.nbin, meta.grid_res,
                    meta.img_len_ms, meta.phase_mode)
            if log_every_s and i % int(log_every_s / sim.dt) == 0:
                print(f"skysim: image {i}/{n_total}, {len(sim.truth)} FRBs so far, "
                      f"max lateness {late:.2f} s", flush=True)
    finally:
        w.close()
    return dict(images=n_total, frbs=len(sim.truth), max_late_s=late)


def make_frb_list(start, duration, rate_per_hour, dm=(50.0, 600.0), snr=(10.0, 30.0),
                  f_low=33.0, f_high=77.3, el_min=20.0, margin_s=400.0, seed=None,
                  spectra=False):
    """
    FRBs for several bands at once: every burst reaches the top of the highest
    band after start + 120 s and the bottom of the lowest band before
    start + duration - margin_s, at a position above el_min for the whole
    sweep. Returns a list of dicts (id, ra, dec, dm, t_inf, snr, width_s).
    """
    rng = np.random.default_rng(seed)
    n = rng.poisson(rate_per_hour * duration / 3600.0)
    out = []
    for _ in range(50 * max(n, 1)):
        if len(out) >= n:
            break
        d = float(math.exp(rng.uniform(math.log(dm[0]), math.log(dm[1]))))
        lo = start + 120.0 - K_DM * d / f_high ** 2
        hi = start + duration - margin_s - K_DM * d / f_low ** 2
        if hi <= lo:
            continue
        t_inf = float(rng.uniform(lo, hi))
        t_top, t_bot = t_inf + K_DM * d / f_high ** 2, t_inf + K_DM * d / f_low ** 2
        ra = float(rng.uniform(0, 360))
        dec = float(np.degrees(np.arcsin(rng.uniform(-0.7, 1.0))))
        if altaz(ra, dec, t_top)[0] < el_min or altaz(ra, dec, t_bot)[0] < el_min:
            continue
        frb = dict(id="sim-" + uuid.uuid4().hex[:10], ra=ra, dec=dec, dm=d, t_inf=t_inf,
                   snr=float(rng.uniform(*snr)),
                   width_s=float(math.exp(rng.uniform(math.log(0.001), math.log(0.05)))))
        if spectra:
            frb.update(random_spectrum(rng, f_low, f_high))
        out.append(frb)
    return sorted(out, key=lambda f: f["t_inf"])


def random_spectrum(rng, f_low, f_high):
    """
    A spectrum and scattering for a simulated burst:
      40% a power law, index -3..+1 (typical of FRBs);
      20% a steep power law, index -8..+8;
      40% emission confined to a Gaussian 5-30% wide, centred within or just
          outside the observed range (as repeaters often show).
    Scattering at 1 GHz is log-uniform between 10 ns and 3 us, which is 0.007-2 s
    at 34 MHz and 0.3-90 ms at 76 MHz.
    """
    u = rng.random()
    if u < 0.4:
        sp = dict(kind="power_law", alpha=float(rng.uniform(-3, 1)), nu_ref=60.0)
    elif u < 0.6:
        sp = dict(kind="power_law", alpha=float(rng.uniform(-8, 8)), nu_ref=60.0)
    else:
        sp = dict(kind="gaussian", nu0=float(rng.uniform(0.8 * f_low, 1.1 * f_high)),
                  frac=float(rng.uniform(0.05, 0.3)))
    return dict(spectrum=sp, tau_1ghz=float(math.exp(rng.uniform(math.log(1e-8), math.log(3e-6)))))


def spectral_factor(sp, nu):
    """S/N at frequency nu relative to the listed S/N (1 without a spectrum)"""
    if not sp:
        return 1.0
    if sp["kind"] == "power_law":
        return float((nu / sp.get("nu_ref", 60.0)) ** sp["alpha"])
    if sp["kind"] == "gaussian":
        return float(math.exp(-0.5 * ((nu - sp["nu0"]) / (sp["frac"] * sp["nu0"])) ** 2))
    raise ValueError(sp["kind"])
