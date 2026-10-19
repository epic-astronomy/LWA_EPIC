"""
Live test injections.

At random times (Poisson, mean interval_s) the injector adds a dispersed burst
to the incoming images, before they are regridded, at a random tracked
position that stays above the elevation limit for the whole sweep. Its DM is
drawn log-uniformly over the range the position can still complete, and its
amplitude is set from the measured noise of that pixel for a target S/N. Each
injection is recorded in <out_dir>/injections.jsonl when it is scheduled and
again when it is recovered or missed, so the detection efficiency of the
running system is measured continuously.

Injections exercise everything after the imager: regridding, normalisation,
RFI excision, dedispersion, search, clustering and storage.

Scattering. By default a burst is scattered as FRBs at these frequencies are:
tau at 600 MHz is drawn log-uniformly over tau600_ms (0.31-10 ms, the base
distribution of the sensitivity study) and scaled as nu^-4, a one-sided
exponential in every channel (burst.channel_fractions); at 57-76 MHz that is
1.4-120 s. A fraction unscattered_fraction is injected unscattered, as a check
of short (pulsar-like) pulses. The target S/N is that of an ideal search for
the scattered burst (its width added in quadrature, burst.effective_samples),
so the recovered over the target S/N measures the search efficiency for real
bursts. tau_1ghz is recorded with each injection, and tau_s at the band centre.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid

import numpy as np

from .burst import Burst, amplitude_for_snr, channel_fractions, effective_samples, stamp
from .candidates import separation_deg
from .dedisp import K_DM
from .geometry import beam_fwhm_deg, time_to_set


class LiveInjector:
    def __init__(self, out_dir: str, interval_s: float = 3600.0, snr=(10.0, 30.0),
                 dm=(10.0, 3000.0), width_s=(0.001, 0.05), lead_s: float = 5.0,
                 rng=None, enabled: bool = True, scatter: bool = True,
                 tau600_ms=(0.31, 10.0), unscattered_fraction: float = 0.2):
        self.path = os.path.join(out_dir, "injections.jsonl")
        os.makedirs(out_dir, exist_ok=True)
        self.interval = interval_s
        self.snr = snr
        self.dm = dm
        self.width = width_s
        self.lead = lead_s
        self.scatter = scatter
        self.tau600_ms = tau600_ms
        self.unscattered_fraction = unscattered_fraction
        self.rng = rng or np.random.default_rng()
        self.enabled = enabled  # random injections need interval_s > 0 as well
        self.next_at = None
        self.active: list[Burst] = []
        self.waiting: list[Burst] = []  # requested from outside, not yet started
        self.done: list[dict] = []
        self.lock = threading.Lock()
        self.stats = dict(injected=0, recovered=0, missed=0)

    def _log(self, rec: dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def schedule(self, now: float, grid, searcher, min_age_images: int):
        """Possibly start a new injection at image time now"""
        if not self.enabled:
            return None
        if self.interval <= 0:
            return None
        if self.next_at is None:
            self.next_at = now + self.rng.exponential(self.interval)
        if now < self.next_at:
            return None
        self.next_at = now + self.rng.exponential(self.interval)
        slots = np.nonzero((grid.hpix_of >= 0) & (searcher.age >= min_age_images))[0]
        if len(slots) == 0:
            return None
        edges = searcher.edges
        sweep = K_DM * (edges[0] ** -2 - edges[-1] ** -2)
        tau1 = self.draw_tau_1ghz()
        tail = 7.0 * tau1 * (float(edges[0]) / 1000.0) ** -4
        for _ in range(20):
            s = int(self.rng.choice(slots))
            h = int(grid.hpix_of[s])
            ra, dec = float(grid.ra[h]), float(grid.dec[h])
            left = float(time_to_set(dec, ra, now, grid.el_min))
            dm_hi = min(self.dm[1], (left - self.lead - 60.0 - tail) / sweep,
                        searcher.tiers[-1].dm_max)
            if dm_hi > self.dm[0]:
                break
        else:
            return None
        dm = float(math.exp(self.rng.uniform(math.log(self.dm[0]), math.log(dm_hi))))
        t_top = now + self.lead
        t_inf = t_top - K_DM * dm / edges[-1] ** 2
        width = float(math.exp(self.rng.uniform(math.log(self.width[0]), math.log(self.width[1]))))
        snr = float(self.rng.uniform(*self.snr))
        b = Burst(id="inj-" + uuid.uuid4().hex[:10], ra=ra, dec=dec, dm=dm, t_inf=t_inf,
                  width_s=width, snr=snr, kind="injection", tau_1ghz=tau1)
        b.meta = dict(scheduled_unix=now)
        self._amplitude(b, s, searcher)
        with self.lock:
            self.active.append(b)
        self.stats["injected"] += 1
        self._log(dict(event="scheduled", **b.to_dict()))
        return b

    def draw_tau_1ghz(self) -> float:
        """
        Scattering time at 1 GHz (s) of a new injection: 0 for a fraction
        unscattered_fraction (or with scatter off), else tau at 600 MHz
        log-uniform over tau600_ms, scaled as nu^-4
        """
        if not self.scatter or self.rng.random() < self.unscattered_fraction:
            return 0.0
        lo, hi = self.tau600_ms
        t600 = 1e-3 * math.exp(self.rng.uniform(math.log(lo), math.log(hi)))
        return float(t600 * (1000.0 / 600.0) ** -4)

    def _amplitude(self, b: Burst, slot: int, searcher):
        """Set b's per-channel amplitude from the noise of the tracked pixel slot"""
        edges = searcher.edges
        tier = searcher.tier_for_dm(b.dm)
        n_eff = effective_samples(b, edges, tier.dt)
        b.amp_sigma = amplitude_for_snr(b.snr, searcher.C, n_eff)
        # per-channel amplitude in image units: a per-image noise sigma of this pixel
        sig = np.sqrt(np.maximum(searcher.var[:, slot], 0)).astype(np.float64)
        b._amp = b.amp_sigma * sig * tier.decim ** 0.5  # noise of a tier sample is decim x
        span = b.span(edges)
        tau_c = float(b.tau(0.5 * (edges[0] + edges[-1]))) if b.tau_1ghz > 0 else 0.0
        b.meta.update(slot=int(slot), tier=tier.level, n_eff=n_eff,
                      sigma=float(np.median(sig)), span=list(span), tau_s=tau_c,
                      expire=span[1] + 3 * tier.block * tier.dt + 120.0)

    def add_external(self, spec: dict) -> Burst:
        """
        A burst requested from outside (the combiner's coordinated injections):
        the same position, DM and time in every band, each band setting the
        amplitude for the requested S/N from its own noise once the position is
        tracked. Fields: id, ra, dec, dm, t_inf, and optional snr, width_s,
        tau_1ghz (unscattered unless given).
        """
        for k in ("id", "ra", "dec", "dm", "t_inf"):
            if k not in spec:
                raise ValueError(f"injection needs {k}")
        b = Burst(id=str(spec["id"]), ra=float(spec["ra"]) % 360.0, dec=float(spec["dec"]),
                  dm=float(spec["dm"]), t_inf=float(spec["t_inf"]),
                  width_s=float(spec.get("width_s", 0.005)), snr=float(spec.get("snr", 20.0)),
                  kind="injection", tau_1ghz=float(spec.get("tau_1ghz", 0.0)))
        b.meta = dict(external=True, received_unix=time.time())
        with self.lock:
            if any(o.id == b.id for o in self.active + self.waiting):
                return b
            self.waiting.append(b)
        self._log(dict(event="requested", **b.to_dict()))
        return b

    def prepare(self, now: float, grid, searcher):
        """Start requested injections whose position is tracked; drop those that passed"""
        if not self.waiting:
            return
        edges = searcher.edges
        with self.lock:
            waiting, self.waiting = self.waiting, []
        for b in waiting:
            if b.arrival(edges[-1]) < now:  # it should already have started here
                self._log(dict(event="not injected", id=b.id, reason="arrived before it could start"))
                continue
            s = grid.nearest_slot(b.ra, b.dec)
            if s < 0 or searcher.age[s] <= searcher.warmup:
                with self.lock:
                    self.waiting.append(b)  # not tracked (yet)
                continue
            self._amplitude(b, s, searcher)
            with self.lock:
                self.active.append(b)
            self.stats["injected"] += 1
            self._log(dict(event="scheduled", **b.to_dict()))

    def apply(self, image: np.ndarray, t0: float, dt: float, edges, centres, wcs, pix_deg):
        """Add the active injections to image [C][y][x] for the image [t0, t0 + dt)"""
        with self.lock:
            active = list(self.active)
        for b in active:
            span = b.meta["span"]
            if t0 + dt < span[0] or t0 > span[1]:
                continue
            frac = channel_fractions(b, edges, t0, dt)
            if not frac.any():
                continue
            x, y = wcs.all_world2pix(b.ra, b.dec, 1, ra_dec_order=True)
            # images are integrated over dt: a fluence F adds F / dt to the mean,
            # and the amplitude was set in units of the per-image noise
            stamp(image, b, float(x), float(y), b._amp, frac, centres, pix_deg)

    def apply_pixels(self, raw: np.ndarray, xy: np.ndarray, t0: float, dt: float, edges, centres,
                     wcs, pix_deg):
        """
        Add the active injections to the pixel spectra raw [C][P] (the imager
        regridded the image on the GPU): the beam stamp of apply, evaluated at
        each slot's image position xy [P, 2] (1-based FITS pixels) for this
        image, which is what regridding the stamped image gives for a beam
        much wider than a pixel.
        """
        with self.lock:
            active = list(self.active)
        for b in active:
            span = b.meta["span"]
            if t0 + dt < span[0] or t0 > span[1]:
                continue
            frac = channel_fractions(b, edges, t0, dt)
            act = np.nonzero(frac > 0)[0]
            if len(act) == 0:
                continue
            x, y = wcs.all_world2pix(b.ra, b.dec, 1, ra_dec_order=True)
            sig = beam_fwhm_deg(float(np.max(centres[act]))) / pix_deg / 2.3548
            r = 3.0 * beam_fwhm_deg(float(np.min(centres[act]))) / pix_deg
            d2 = (xy[:, 0] - float(x)) ** 2 + (xy[:, 1] - float(y)) ** 2
            near = np.nonzero(np.isfinite(d2) & (d2 <= r * r))[0]
            if len(near) == 0:
                continue
            for c in act:
                s = sig * centres.max() / centres[c]  # beam scales with wavelength
                raw[c, near] += (b._amp[c] * frac[c]) * np.exp(-d2[near] / (2 * s * s))

    def match(self, cand, radius_deg: float):
        """Return the active injection a candidate corresponds to, if any"""
        with self.lock:
            for b in self.active:
                if separation_deg(b.ra, b.dec, cand.ra, cand.dec) > radius_deg:
                    continue
                t_bot = b.t_inf + K_DM * b.dm / cand.f_lo_mhz ** 2
                tau = 3.0 * float(b.tau(cand.f_lo_mhz)) if b.tau_1ghz > 0 else 0.0
                if abs(t_bot - cand.t_bottom) > 3 * cand.tier_dt + 2 * cand.width_s + 2.0 + tau:
                    continue
                # a wide boxcar (or a scattering tail) fixes the DM only to its
                # width over the sweep per DM
                ddm = (cand.width_s + tau) / (K_DM * (cand.f_lo_mhz ** -2 - cand.f_hi_mhz ** -2))
                if abs(b.dm - cand.dm) > max(0.1 * b.dm, 3 * cand.dm_step, ddm):
                    continue
                return b
        return None

    def record(self, b: Burst, cand):
        if "recovered" not in b.meta or cand.snr > b.meta["recovered"]["snr"]:
            first = "recovered" not in b.meta
            b.meta["recovered"] = dict(cand_id=cand.id, snr=cand.snr, dm=cand.dm,
                                       t_inf=cand.t_inf, ra=cand.ra, dec=cand.dec,
                                       tier=cand.tier)
            if first:
                self.stats["recovered"] += 1

    def expire(self, now: float):
        with self.lock:
            keep = []
            for b in self.active:
                if now > b.meta["expire"]:
                    rec = dict(event="recovered" if "recovered" in b.meta else "missed",
                               id=b.id, dm=b.dm, snr_target=b.snr, ra=b.ra, dec=b.dec,
                               t_inf=b.t_inf, tau_1ghz=b.tau_1ghz,
                               tau_s=b.meta.get("tau_s", 0.0), **({"found": b.meta["recovered"]}
                                                if "recovered" in b.meta else {}))
                    if rec["event"] == "missed":
                        self.stats["missed"] += 1
                    self.done.append(rec)
                    self.done = self.done[-500:]
                    self._log(rec)
                else:
                    keep.append(b)
            self.active = keep

    def summary(self):
        s = dict(self.stats)
        s["active"] = len(self.active)
        with self.lock:  # the injections still in progress, for comparing candidates with them
            s["in_progress"] = [dict(id=b.id, ra=b.ra, dec=b.dec, dm=b.dm, t_inf=b.t_inf,
                                     snr_target=b.snr, width_s=b.width_s) for b in self.active]
        s["next_at"] = self.next_at
        s["recent"] = self.done[-50:]
        return s
