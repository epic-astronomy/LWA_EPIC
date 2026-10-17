"""
The all-sky monitor of one EPIC imager.

A monitor follows the images of the imager on its own node (shared-memory
ring, shm.py), tracks every HEALPix pixel above the elevation limit, searches
each for dispersed bursts in real time (search.py) and stores what it finds
(candidates.py). It needs nothing from any other node: each worker runs its
own monitor for its own band, and the loss of a worker only removes that band.

Every 10 s of images the monitor also records the data quality of its band
(conditions.py): noise relative to the last hour, RFI and trigger rates,
missing images, and whether the period is disturbed. Each candidate carries
the summary of the minute before it.

Images are processed in time order. A missing image (packet loss, a gap in
the ring) is searched as zeros, so the time axis stays regular; a gap longer
than max_gap_s restarts the search state.
"""
from __future__ import annotations

import collections
import json
import math
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from . import kernels
from .accumulate import Accumulator
from .conditions import Conditions
from .candidates import (Candidate, CandidateStore, cluster_hits, elevation_radius,
                         separation_deg, utc)
from .dedisp import K_DM, fdmt, fdmt_plan
from .geometry import Band, SkyGrid, altaz, auto_nside, beam_fwhm_deg, image_wcs
from .inject import LiveInjector
from .known import KnownSources
from .search import Searcher
from .flux import (CalibrationFeed, beam_sigma_pix, beam_table, candidate_flux, measure_calibrators,
                   peak_fit)
from .quality import candidate_quality, channel_contributions, contribution_noise
from .watch import Watches, measure, predicted_bottom


@dataclass
class Config:
    out_dir: str = "skymon_out"  # on local disk: candidates, spool, logs (never the archive)
    db_dsn: str | None = None  # the node's local PostgreSQL (localdb.py); None: files only
    nside: int = 0  # 0: from the beam at the band centre
    el_min: float = 10.0
    dm_max: float = 3000.0
    threshold: float = 8.0
    widths: tuple = (1, 2, 4, 8)
    block0: int = 256
    nthreads: int = 8
    regrid_interval_s: float = 10.0
    max_gap_s: float = 60.0
    warmup_images: int = 64
    norm_tau_s: float = 120.0
    var_tau_s: float = 300.0
    hp_tau: float = 32.0
    # a normalised cell beyond clip is set to zero unless at least
    # clip_sky_neighbours of its sky neighbours deviate the same way by
    # clip_sky_fraction of it (a source seen through the beam). A burst whose
    # channel signal falls into one image adds S/N sqrt(n_eff / nchan) to that
    # cell (n_eff about 1.7): 12 at S/N 73 for 64 channels, 104 for 128
    # (search.Searcher)
    clip: float = 12.0
    clip_sky_neighbours: int = 3
    clip_sky_fraction: float = 0.3
    chan_flag_sigma: float = 1.6  # spread over the sky above which a channel is flagged
    flag_tau_s: float = 30.0  # time constant of the flag rate of each channel
    mask_on: float = 0.5  # a channel flagged at this rate is masked ...
    mask_off: float = 0.2  # ... until its rate falls below this
    accum_interval_s: float = 10.0
    accum_hours: float = 6.0
    rfi_max_beams: float = 6.0  # a cluster wider than this many beam areas is RFI
    rfi_sky_fraction: float = 0.01  # a moment that triggers this much of the sky is RFI
    dm_min: float = 2.0  # candidates below this DM are classified as RFI (terrestrial)
    rfi_min_sites: int = 4  # clusters at this many separate positions at once are RFI
    inject_interval_s: float = 3600.0  # 0 disables live injections
    inject_snr: tuple = (10.0, 30.0)
    inject_dm: tuple = (10.0, 3000.0)
    store_rfi: bool = True
    # a candidate whose signal is in fewer effective channels than this, or
    # with more than narrowband_top2 of it in two channels, is narrowband RFI
    narrowband_min_channels: float = 6.0
    narrowband_top2: float = 0.6
    known_pulsars_csv: str | None = None  # name,ra_deg,dec_deg,dm,period_s: extra pulsars
    # data-quality periods (conditions.py), sampled on the image clock
    calinfo_path: str | None = None  # streamcal's calibration record (flux.py); None: no flux
    flux_every_s: float = 60.0  # calibrators measured in the images this often
    beam_model: str = "empirical"  # lsl dipole model for the primary beam; "record": the record's
    cond_interval_s: float = 10.0
    cond_baseline_tau_s: float = 3600.0  # slow baseline of the noise
    cond_hours: float = 6.0  # samples kept in memory
    cond_noise_ratio: float = 1.5  # disturbed above this noise over its baseline
    cond_rfi_rate: float = 2.0  # disturbed above this many RFI candidates per minute
    cond_sky_fraction: float = 0.01  # disturbed above this fraction of the sky with hits
    cond_npix: int = 1024  # pixels in the median of the noise ratio
    seed: int | None = None
    host: str = field(default_factory=socket.gethostname)


class Monitor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # process() runs in one thread; queries take only grid_lock, which
        # guards the slot assignment and is held for milliseconds, so the
        # HTTP interface answers while a search runs
        self.lock = threading.RLock()
        self.grid_lock = threading.Lock()
        self.rng = np.random.default_rng(cfg.seed)
        os.makedirs(cfg.out_dir, exist_ok=True)
        from .localdb import LocalDB
        self.db = LocalDB(cfg.db_dsn, os.path.join(cfg.out_dir, "spool")) if cfg.db_dsn else None
        self.store = CandidateStore(cfg.out_dir, db=self.db)
        self.injector = LiveInjector(cfg.out_dir, cfg.inject_interval_s, cfg.inject_snr,
                                     cfg.inject_dm, rng=self.rng)
        self.watches = Watches()
        self.cond = Conditions(cfg.cond_interval_s, cfg.cond_baseline_tau_s, cfg.cond_hours,
                               cfg.cond_noise_ratio, cfg.cond_rfi_rate, cfg.cond_sky_fraction,
                               cfg.cond_npix, rng=np.random.default_rng(cfg.seed))
        self._disturbed = False
        # calibration: the gain table version of the images (epic++ --gain_table,
        # streamcal), and its changes over the last 6 h as (unix, old, new)
        self.gain_version = None
        self.gain_changes = collections.deque(maxlen=20000)
        # flux calibration: streamcal's records, and the calibrators measured in the images
        self.calfeed = CalibrationFeed(cfg.calinfo_path)
        self.flux_meas = None
        self.flux_unwritten = collections.deque(maxlen=2000)
        self._flux_next = None
        self.band = None
        self.searcher = None
        self.grid = None
        self.accum = None
        self.wcs = None
        self.seq0 = None
        self.last_index = -1
        self.started = time.time()
        self.metrics = collections.defaultdict(float)
        self.lag_hist = collections.deque(maxlen=600)
        self.last_image_unix = None
        self.config_key = None

    # ------------------------------------------------------------ setup
    def _setup(self, meta):
        cfg = self.cfg
        self.band = Band(meta.chan0, meta.nchan, meta.nbin)
        self.dt = meta.img_len_ms / 1e3
        self.nseqs = meta.nseqs
        self.seq0 = meta.seq_start
        self.unix0 = meta.unix
        nside = cfg.nside or auto_nside(self.band.f_centre, meta.grid_res)
        self.grid = SkyGrid(nside, cfg.el_min, meta.grid, meta.grid_res, meta.phase_mode)
        self.beam_tab = None  # the primary beam of this band (flux.py)
        if cfg.beam_model != "record":
            try:
                e = self.band.edges_mhz()
                self.beam_tab = beam_table([e[0], self.band.f_centre, e[-1]], cfg.beam_model)
            except Exception as ex:  # no lsl: the calibration record's beam, if any
                print(f"skymon: primary beam ({cfg.beam_model}) unavailable: {ex!r}", flush=True)
        self.searcher = Searcher(self.band.edges_mhz(), self.dt, self.grid.capacity,
                                 dm_max=cfg.dm_max, widths=cfg.widths, block0=cfg.block0,
                                 threshold=cfg.threshold, nthreads=cfg.nthreads,
                                 warmup_images=cfg.warmup_images, norm_tau_s=cfg.norm_tau_s,
                                 var_tau_s=cfg.var_tau_s, hp_tau=cfg.hp_tau, clip=cfg.clip,
                                 unix0=self.unix0, rng=self.rng,
                                 chan_flag_sigma=cfg.chan_flag_sigma, flag_tau_s=cfg.flag_tau_s,
                                 mask_on=cfg.mask_on, mask_off=cfg.mask_off,
                                 clip_nbr_min=cfg.clip_sky_neighbours,
                                 clip_nbr_frac=cfg.clip_sky_fraction)
        self.accum = Accumulator(self.band.nchan, self.grid.capacity, self.dt,
                                 cfg.accum_interval_s, cfg.accum_hours)
        self.raw = np.zeros((self.band.nchan, self.grid.capacity), np.float32)
        self.beam = beam_fwhm_deg(self.band.f_centre)
        self.radius = max(2.0 * self.beam, 2.5 * self.grid.pix_deg)
        try:
            self.known = KnownSources(self.beam, cfg.known_pulsars_csv)
        except (OSError, ValueError, KeyError) as e:
            print(f"skymon: could not read {cfg.known_pulsars_csv} ({e}); "
                  f"using the built-in pulsars", flush=True)
            self.known = KnownSources(self.beam)
        beam_area = 1.1331 * self.beam ** 2
        self.rfi_max_pix = max(9, int(cfg.rfi_max_beams * beam_area / self.grid.pix_deg ** 2))
        self.last_index = -1
        self.grid_next = None
        self.config_key = (meta.chan0, meta.nchan, meta.nbin, meta.grid, meta.grid_res,
                           meta.nseqs, meta.phase_mode)
        self.wcs_unix = None
        self.cond.restart((self.band.nchan, self.grid.capacity))
        print(f"skymon: band {self.band.f_lo:.3f}-{self.band.f_hi:.3f} MHz, {self.band.nchan} ch, "
              f"dt {self.dt * 1e3:.1f} ms, nside {nside} ({self.grid.pix_deg:.2f} deg), "
              f"capacity {self.grid.capacity} px, beam {self.beam:.2f} deg", flush=True)
        for t in self.searcher.describe():
            print(f"skymon:   tier {t['level']}: dt {t['dt_s']:.3f} s, DM {t['dm_min']:.1f}-"
                  f"{t['dm_max']:.1f} step {t['dm_step']:.2f}, latency {t['latency_s']:.0f} s",
                  flush=True)
        mem = self.searcher.memory_bytes() + self.accum.memory_bytes()
        print(f"skymon: state memory {mem / 2 ** 30:.1f} GiB", flush=True)

    def _update_grid(self, unix):
        with self.grid_lock:
            added, removed = self.grid.update(unix, self.cfg.regrid_interval_s)
            # state starts zeroed, so only a slot whose pixel set needs clearing
            # before another pixel can rise into it
            self.searcher.reset_slots(removed)
            self.accum.reset_slots(removed)
            self.cond.reset_slots(removed)
        self.searcher.set_live(self.grid.active())
        self.searcher.set_neighbours(self.grid.nbr)
        self.wcs = self.grid.wcs
        self.wcs_unix = unix
        self.grid_next = unix + self.cfg.regrid_interval_s
        self.metrics["tracked"] = int(self.grid.active().sum())

    # ------------------------------------------------------------ per image
    def process(self, meta, image):
        """Process one image (meta: shm.ImageMeta, image [C][y][x] float32, modified)"""
        key = (meta.chan0, meta.nchan, meta.nbin, meta.grid, meta.grid_res, meta.nseqs,
               meta.phase_mode)
        with self.lock:
            if self.config_key != key:
                if self.config_key is not None:
                    print("skymon: imaging configuration changed, restarting the search",
                          flush=True)
                self._setup(meta)
            n = int(round((meta.seq_start - self.seq0) / self.nseqs))
            if n <= self.last_index:
                if self.last_index - n > 10:  # time went backwards: restart
                    self._setup(meta)
                    n = 0
                else:
                    self.metrics["late_images"] += 1
                    return []
            gap = n - self.last_index - 1
            if gap * self.dt > self.cfg.max_gap_s:
                print(f"skymon: {gap * self.dt:.0f} s gap, restarting the search", flush=True)
                self._setup(meta)
                n = 0
                gap = 0
            out = []
            for _ in range(gap):
                self.metrics["missing_images"] += 1
                out += self._push(None, self.last_index + 1)
            out += self._push((meta, image), n)
            self.last_image_unix = meta.unix
            return out

    def _push(self, item, n):
        t0 = time.perf_counter()
        unix = self.unix0 + n * self.dt
        if self.cond.due(unix):
            self._sample_conditions()
        self.cond.note_image(item is None)
        if self.grid_next is None or unix >= self.grid_next:
            self._update_grid(unix)
        else:
            self.grid.taps(unix)
        self.last_index = n
        if item is None:
            results = self.searcher.push(None)
        else:
            meta, image = item
            self._note_gains(meta.gain_version, unix)
            self.injector.schedule(unix, self.grid, self.searcher,
                                   self.cfg.warmup_images + 4 * self.cfg.block0)
            self.injector.prepare(unix, self.grid, self.searcher)
            if self.injector.active:
                self.injector.apply(image, unix, self.dt, self.band.edges_mhz(),
                                    self.band.centres_mhz(), self.wcs, self.cfg_pix_deg(meta))
            kernels.regrid(image, self.grid.idx, self.grid.w, self.raw, self.cfg.nthreads)
            self.accum.begin(unix)
            results = self.searcher.push(self.raw, self.accum.sum)
            self.accum.end_image(unix + self.dt)
        found = []
        for res in results:
            self.cond.note_block(res)
            found += self._candidates(res)
            found += self._measure_watches(res)
        self._expire_watches(unix)
        self.searcher.finish_block()
        self.injector.expire(unix)
        self.metrics["images"] += 1
        self.metrics["process_s"] += time.perf_counter() - t0
        return found

    def _note_gains(self, version, unix):
        """
        Follow the calibration of the images. streamcal updates the gains about
        every 10 s, which changes each image only slightly; applying the first
        table, or reverting to the nominal phases, changes the images more.
        No image is held back: the changes are counted in the conditions and
        recorded with every candidate whose sweep they fall in, for review.
        """
        version = int(version)
        old, self.gain_version = self.gain_version, version
        if old is None or version == old:
            return
        self.gain_changes.append((unix, old, version))
        self.cond.note_gain_change()
        if old == 0:
            print(f"skymon: calibration applied (gain table v{version}) at {utc(unix)}", flush=True)
        elif version == 0:
            print(f"skymon: calibration reverted to the nominal phases at {utc(unix)}", flush=True)

    def _flux_calibration(self, unix):
        """Read streamcal's record, and measure the calibrators in the images every flux_every_s"""
        try:
            self.calfeed.poll()
            if self._flux_next is not None and unix < self._flux_next:
                return
            self._flux_next = unix + self.cfg.flux_every_s
            rec = self.calfeed.for_version(self.gain_version)
            if rec is None:
                return
            with self.grid_lock:
                m = measure_calibrators(self.grid, self.accum, self.beam, rec,
                                        self.band.centres_mhz(), unix,
                                        beam_tab=getattr(self, "beam_tab", None))
            if m is not None:
                self.flux_meas = m
                self.flux_unwritten.append(m)
        except Exception as e:  # flux is for review; it must never stop the search
            print(f"skymon: flux calibration failed: {e!r}", flush=True)

    def _peak(self, window, slot, row, t_idx, width):
        """
        peak_fit over a candidate's pixel and its 8 neighbours: each pixel's
        contributions along the track (high-pass undone), summed over the band,
        against its image position; the beam's width at the band centre. The
        fitted position is converted to RA, Dec with the grid's WCS. Returns
        peak_fit's dict with ra_deg, dec_deg added, or None on failure.
        """
        try:
            with self.grid_lock:
                hp_c = int(self.grid.hpix_of[slot])
                nb = self.grid.neighbours(hp_c) if hp_c >= 0 else []
                slots = [slot] + [int(self.grid.slot_of[n]) for n in nb]
                xy = np.array([self.grid.xy[q] if q >= 0 else (np.nan, np.nan) for q in slots])
                xy_a = self.grid.xy_a[slot].copy()
                wcs = self.grid.wcs
            if any(q < 0 for q in slots):
                vals = np.full(len(slots), np.nan)
            else:
                e = self.band.edges_mhz()
                a = self.searcher.a_hp
                vals = np.array([float(np.sum(channel_contributions(window[:, :, q], e, row, t_idx,
                                                                    width, a_hp=a)))
                                 for q in slots])
            sig = float(beam_sigma_pix([self.band.f_centre], self.grid.grid_res)[0])
            out = peak_fit(xy, vals, sig)
            out["n_pixels"] = len(slots)
            if out["x"] is not None:
                # the fitted offset, applied to the pixel's position at the WCS's time
                px = xy_a[0] + (out["x"] - xy[0, 0])
                py = xy_a[1] + (out["y"] - xy[0, 1])
                ra, dec = wcs.all_pix2world(px, py, 1, ra_dec_order=True)
                out.update(ra_deg=round(float(ra) % 360.0, 4), dec_deg=round(float(dec), 4))
            return out
        except Exception as ex:  # flux is for review; it must never stop the search
            print(f"skymon: peak fit failed: {ex!r}", flush=True)
            return None

    def _flux(self, contrib, raw_sigma, tier_level, width_s, el, az, snr, contrib_err=None,
              peak=None):
        """Fluence of a candidate from the latest calibrator measurement (None without one)"""
        m = self.flux_meas
        if m is None or m.get("k") is None:
            return None
        rec = self.calfeed.for_version(self.gain_version)
        try:
            f = candidate_flux(contrib, raw_sigma, tier_level, self.dt, width_s, m["k"], rec, el, az,
                               self.band.centres_mhz(), snr,
                               beam_tab=getattr(self, "beam_tab", None), contrib_err=contrib_err,
                               peak=peak)
        except Exception as e:
            print(f"skymon: flux of a candidate failed: {e!r}", flush=True)
            return None
        if f is not None:
            f.update(calibrators=[dict(name=c["name"], el_deg=c["el_deg"], excess_snr=c["excess_snr"],
                                       scale=c["scale"]) for c in m["calibrators"]],
                     measured_unix=m["t"], measured_gain_version=m["gain_version"])
        return f

    def _calibration_around(self, c):
        """Gain table of a candidate and the changes within its sweep (with a margin of its width)"""
        t0, t1 = c.t_top - 2 * c.width_s, c.t_bottom + c.width_s
        ch = [g for g in self.gain_changes if t0 <= g[0] <= t1]
        return dict(gain_version=self.gain_version, changes_in_sweep=len(ch),
                    applied_or_reverted_in_sweep=any(o == 0 or n == 0 for _, o, n in ch))

    def calibration_status(self, window_s=3600.0):
        now = self.last_image_unix or time.time()
        recent = [g for g in self.gain_changes if g[0] > now - window_s]
        last = self.gain_changes[-1] if self.gain_changes else None
        return dict(gain_version=self.gain_version, calibrated=bool(self.gain_version),
                    changes_last_hour=len(recent),
                    last_change=None if last is None else dict(t=last[0], old=last[1], new=last[2]))

    def cfg_pix_deg(self, meta):
        return meta.grid_res

    def _sample_conditions(self):
        """
        Close a conditions interval (every cond_interval_s of images). The
        sample is only queued for the database, which the database thread
        writes (sync_db), so the search never waits for it.
        """
        s = self.searcher
        rec = self.cond.sample(s.var, s.age, s.live, s.warmup, self.metrics["tracked"])
        self._flux_calibration(rec["t"])
        was, self._disturbed = self._disturbed, rec["disturbed"]
        if rec["disturbed"] and not was:
            print(f"skymon: disturbed period from {utc(rec['t'] - rec['interval_s'])} "
                  f"({', '.join(rec['reasons'])}; noise ratio {rec['noise_ratio'] or 0:.2f}, "
                  f"RFI {rec['rfi_rate']:.1f}/min, sky triggered "
                  f"{rec['sky_fraction_triggered']:.3f})", flush=True)
        elif was and not rec["disturbed"]:
            print(f"skymon: disturbed period ended at {utc(rec['t'] - rec['interval_s'])}",
                  flush=True)

    # ------------------------------------------------------------ candidates
    def _candidates(self, res):
        tier = res.tier
        if res.calibrating or not res.hits:
            return []
        hits = res.hits
        hp = self.grid.hpix_of[[h.slot for h in hits]]
        ok = hp >= 0
        hits = [h for h, k in zip(hits, ok) if k]
        hp = hp[ok]
        if not hits:
            return []
        ra, dec = self.grid.ra[hp], self.grid.dec[hp]
        el_hits, _ = altaz(ra, dec, hits[0].t_bottom)
        radius = elevation_radius(self.radius, el_hits)
        groups = cluster_hits(hits, ra, dec, radius, self.searcher.D // 4)
        found = []
        n_all = len(hits)
        # moments when much of the sky triggers at once (RFI seen through the
        # sidelobes). A broadband flash at T answers trial delay d at times
        # T..T + d, so each hit stands for the span [t_end - row, t_end]; a
        # span shared by more pixels than a few beams' worth is RFI.
        t_hi = np.array([h.t_end for h in hits]) + 2
        t_lo = np.array([h.t_end - h.row for h in hits]) - 2
        slots = np.array([h.slot for h in hits])
        n_wide = max(3 * self.rfi_max_pix,
                     int(self.cfg.rfi_sky_fraction * max(1, self.metrics["tracked"])))
        busy = np.zeros(len(hits), bool)
        for k in np.argsort(t_hi):
            if busy[k]:
                continue
            ov = (t_lo <= t_hi[k]) & (t_hi >= t_lo[k])
            if len(np.unique(slots[ov])) > n_wide:
                busy |= ov
        # a burst comes from one place: clusters at several well-separated
        # positions with overlapping time spans are one flash of RFI
        heads = [g[0] for g in groups]
        sites = np.zeros(len(groups), int)
        for i, k in enumerate(heads):
            for j, m in enumerate(heads):
                if t_lo[m] <= t_hi[k] and t_hi[m] >= t_lo[k] and (
                        i == j or separation_deg(ra[k], dec[k], ra[m], dec[m]) >
                        2 * max(radius[k], radius[m])):
                    sites[i] += 1
        for gi, g in enumerate(groups):
            b = hits[g[0]]
            flash = bool(busy[g[0]]) or sites[gi] >= self.cfg.rfi_min_sites
            hpix = int(hp[g[0]])
            pix = sorted({int(hp[i]) for i in g})
            ext = float(np.max(separation_deg(ra[g[0]], dec[g[0]], ra[g], dec[g])))
            reasons = []
            if len(pix) > self.rfi_max_pix / np.clip(np.sin(np.radians(el_hits[g[0]])), 0.25, 1) \
                    or ext > 4 * radius[g[0]]:
                reasons.append("wider than a few beams")
            if busy[g[0]]:
                reasons.append("much of the sky at once")
            if sites[gi] >= self.cfg.rfi_min_sites:
                reasons.append("several separate positions at once")
            if b.dm < self.cfg.dm_min:
                reasons.append("DM near zero")
            # certain: terrestrial (no dispersion) or seen over much of the sky
            certain = "DM near zero" in reasons or "much of the sky at once" in reasons
            cls = "rfi" if reasons else "candidate"
            el, az = altaz(ra[g[0]], dec[g[0]], b.t_bottom)
            xy = self.grid.xy[b.slot]
            cid = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime(b.t_inf))}_" \
                  f"{self.band.f_centre:.0f}MHz_{uuid.uuid4().hex[:6]}"
            c = Candidate(
                id=cid, cls=cls, snr=b.snr, dm=b.dm, dm_step=tier.dm_per_delay,
                width_s=b.width_s, t_inf=b.t_inf, t_bottom=b.t_bottom,
                t_top=b.t_bottom - b.row * tier.dt, utc_inf=utc(b.t_inf), hpix=hpix,
                nside=self.grid.nside, ra=float(ra[g[0]]), dec=float(dec[g[0]]), el=float(el),
                az=float(az), image_x=float(xy[0]), image_y=float(xy[1]), tier=tier.level,
                tier_dt=tier.dt, n_pix=len(pix), extent_deg=ext, f_lo_mhz=self.band.f_lo,
                f_hi_mhz=self.band.f_hi, host=self.cfg.host, detected_unix=time.time(),
                members=[[int(hp[i]), round(hits[i].snr, 2)] for i in g[:64]],
                rfi_reasons=reasons, rfi_certain=certain)
            if cls == "rfi" and not self.cfg.store_rfi:
                self.cond.note_candidate(cls)  # counted in the RFI rate though not stored
                continue
            # keep the data of everything that is not certainly RFI (this also
            # measures the candidate's quality)
            products = None if (cls == "rfi" and certain) else self._products(res, b, c)
            if cls != "rfi" and self._narrowband(c.quality):
                # a carrier in one or two channels; not certain, so its data stay.
                # Tested before injections are matched, so the measured efficiency
                # includes any burst this rule would lose
                c.rfi_reasons.append("narrowband (a few channels)")
                cls = c.cls = "rfi"
                if not self.cfg.store_rfi:
                    self.cond.note_candidate(cls)
                    continue
            if cls != "rfi":
                inj = self.injector.match(c, float(elevation_radius(self.radius, c.el)))
                if inj is not None:
                    c.cls = "injection"
                    c.injection_id = inj.id
                    self.injector.record(inj, c)
            if not certain:
                self._label_known(c)
            c.quality["conditions"] = self.cond.around(c.t_bottom)
            c.quality["calibration"] = self._calibration_around(c)
            stored = self.store.add(c, products, self.beam)  # repeats peak within a beam
            if stored is not None:
                self.cond.note_candidate(stored.cls)
                found.append(stored)
                print(f"skymon: {stored.cls} {stored.id} S/N {stored.snr:.1f} DM {stored.dm:.1f} "
                      f"RA {stored.ra:.2f} Dec {stored.dec:.2f} tier {tier.level} "
                      f"({len(pix)} px, {n_all} hits)", flush=True)
        return found

    def _narrowband(self, q: dict) -> bool:
        """Is the signal confined to a few channels (quality.spread_measures)?"""
        n_eff, top2 = q.get("n_eff_channels"), q.get("top2_fraction")
        return (n_eff is not None and n_eff < self.cfg.narrowband_min_channels) or \
            (top2 is not None and top2 > self.cfg.narrowband_top2)

    def _label_known(self, c: Candidate):
        """
        Record a known source at the candidate's position (and DM). A plain
        candidate there becomes class "known": its data are kept, but it opens
        no event. RFI and injections keep their class and only carry the label.
        """
        try:
            ks = self.known.classify(c.ra, c.dec, c.el, c.t_bottom, c.dm, c.dm_step)
        except Exception as e:  # a label for review must never stop the search
            print(f"skymon: known-source check failed: {e!r}", flush=True)
            return
        if ks is None:
            return
        c.known_source = ks
        if c.cls == "candidate":
            c.cls = "known"

    def _products(self, res, hit, cand=None):
        """Data around a candidate: its pixel's window, DM-time plane, and accumulation"""
        tier = res.tier
        s = hit.slot
        win = np.ascontiguousarray(res.window[:, :, s])  # [C, W]
        plan = fdmt_plan(self.band.edges_mhz(), self.searcher.D)
        dmt = fdmt(win[:, :, None], plan)[:, :, 0]
        if cand is not None:
            try:
                cand.quality = candidate_quality(win, dmt, res.sigma, self.band.edges_mhz(),
                                                 hit.row, hit.t_end - res.first_sample,
                                                 hit.width, list(self.searcher.widths))
            except Exception as e:  # a measure for review must never stop the search
                cand.quality = dict(error=repr(e))
            # channels set to zero for interference along the burst's sweep
            fz = self.searcher.tier_zeroed_fraction(tier, hit.t_end - hit.row - hit.width + 1,
                                                    hit.t_end)
            cand.quality["flagged_fraction"] = None if fz is None else round(fz, 4)
        t_win = self.searcher.tier_time(tier, res.first_sample + np.arange(win.shape[1]))
        ta, acc = self.accum.pixel(s, since=hit.t_inf - 1800)
        # the pixel's noise per image in image units: undoes the normalisation (flux.py)
        raw_sigma = np.sqrt(np.maximum(self.searcher.var[:, s], 0)).astype(np.float32)
        raw_mean = self.searcher.mu[:, s].astype(np.float32)
        if cand is not None:
            t_idx = hit.t_end - res.first_sample
            e = self.band.edges_mhz()
            a = self.searcher.a_hp
            contrib = channel_contributions(win, e, hit.row, t_idx, hit.width, a_hp=a)
            err = contribution_noise(win, e, hit.row, t_idx, hit.width, a_hp=a)
            peak = self._peak(res.window, s, hit.row, t_idx, hit.width)
            cand.flux = self._flux(contrib, raw_sigma, tier.level, cand.width_s, cand.el, cand.az,
                                   cand.snr, err, peak)
            if peak is not None:
                cand.quality["peak"] = peak
        rec = self.calfeed.for_version(self.gain_version)
        return dict(dynspec=win.astype(np.float32), dynspec_t=t_win, dmt=dmt.astype(np.float32),
                    dmt_dm=np.arange(self.searcher.D + 1) * tier.dm_per_delay,
                    freqs_mhz=self.band.centres_mhz(), accum=acc, accum_t=ta,
                    sigma=res.sigma, raw_sigma=raw_sigma, raw_mean=raw_mean,
                    tier_images=np.int32(2 ** tier.level), image_s=np.float64(self.dt),
                    calinfo=np.array(json.dumps(rec) if rec else ""),
                    flux_measurement=np.array(json.dumps(self.flux_meas) if self.flux_meas else ""))

    # ------------------------------------------------------------ watches
    def _measure_watches(self, res):
        """Measure the pending watches whose arrival falls in this block of this tier"""
        tier = res.tier
        pend = self.watches.pending()
        if not pend:
            return []
        W = res.window.shape[1]
        first = res.first_sample
        new_lo, new_hi = first + W - tier.block, first + W - 1
        widths = self.searcher.widths
        done = []
        for w in pend:
            if self.searcher.tier_for_dm(w["dm"]) is not tier:
                continue
            tb = predicted_bottom(w, self.band.f_lo)
            s = (tb - self.unix0) / tier.dt - 0.5
            tol = w["t_tol"] / tier.dt + 2 + int(widths.max())
            lo_s, hi_s = int(np.floor(s - tol)), int(np.ceil(s + tol))
            if lo_s > new_hi:
                continue  # still to come
            if hi_s < new_lo and w["best"] is None:
                w.update(state="expired", reason="arrived before the watch could be measured")
                continue
            if res.calibrating:
                w["reason"] = "the tier for this DM was starting"
                if hi_s <= new_hi:
                    w["state"] = "expired"
                continue
            with self.grid_lock:
                h = self.grid.nearest_slot(w["ra"], w["dec"])
                hp_c = int(self.grid.hpix_of[h]) if h >= 0 else -1
                nb = [int(self.grid.slot_of[n]) for n in self.grid.neighbours(hp_c)] \
                    if hp_c >= 0 else []
            slots = [x for x in [h] + nb if x >= 0]
            if not slots:
                w.update(state="expired", reason="position not tracked (below the elevation limit)")
                continue
            dpd = tier.dm_per_delay
            r0 = int(np.floor((w["dm"] - w["dm_err"]) / dpd)) - 1
            r1 = int(np.ceil((w["dm"] + w["dm_err"]) / dpd)) + 1
            t0 = max(lo_s, new_lo) - first
            t1 = min(hi_s, new_hi) - first
            win = res.window[:, :, slots]
            m = measure(win, self.searcher.plan, self.searcher.D, tier, res.sigma, widths,
                        (r0, r1), (t0, t1), zerodm=tier.level > 0)
            if m is not None and (w["best"] is None or m[0] > w["best"]["snr"]):
                snr, row, t, wi, k = m
                wd = int(widths[wi])
                t_end = first + t
                t_bottom = self.searcher.tier_time(tier, t_end - (wd - 1) / 2.0 + 0.5)
                dm = row * dpd
                w["best"] = dict(snr=snr, dm=dm, row=row, t_bottom=t_bottom, width_s=wd * tier.dt,
                                 slot=slots[k], tier=tier.level, tier_dt=tier.dt, dm_step=dpd,
                                 t_inf=t_bottom - K_DM * dm / self.band.f_lo ** 2)
                w["_win"] = np.ascontiguousarray(res.window[:, :, slots[k]])
                w["_raw_sigma"] = np.sqrt(np.maximum(self.searcher.var[:, slots[k]], 0)).astype(
                    np.float32)
                w["_track"] = (row, t, wd)
                w["_peak"] = self._peak(res.window, slots[k], row, t, wd)
                w["_t_win"] = self.searcher.tier_time(tier, first + np.arange(W))
                w["_sigma"] = res.sigma
            if hi_s <= new_hi:
                done.append(self._finish_watch(w))
        return [c for c in done if c is not None]

    def _finish_watch(self, w):
        """Store a measured watch as a followup candidate"""
        b = w["best"]
        if b is None:
            w.update(state="expired", reason=w.get("reason") or "not measured")
            return None
        with self.grid_lock:
            hpix = int(self.grid.hpix_of[b["slot"]])
            xy = self.grid.xy[b["slot"]].copy()
        ra, dec = (float(self.grid.ra[hpix]), float(self.grid.dec[hpix])) if hpix >= 0 \
            else (w["ra"], w["dec"])
        el, az = altaz(ra, dec, b["t_bottom"])
        cid = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime(b['t_inf']))}_" \
              f"{self.band.f_centre:.0f}MHz_fu_{uuid.uuid4().hex[:6]}"
        c = Candidate(
            id=cid, cls="followup", snr=float(b["snr"]), dm=float(b["dm"]), dm_step=b["dm_step"],
            width_s=b["width_s"], t_inf=b["t_inf"], t_bottom=b["t_bottom"],
            t_top=b["t_bottom"] - b["row"] * b["tier_dt"], utc_inf=utc(b["t_inf"]), hpix=hpix,
            nside=self.grid.nside, ra=ra, dec=dec, el=float(el), az=float(az),
            image_x=float(xy[0]), image_y=float(xy[1]), tier=b["tier"], tier_dt=b["tier_dt"],
            n_pix=1, extent_deg=0.0, f_lo_mhz=self.band.f_lo, f_hi_mhz=self.band.f_hi,
            host=self.cfg.host, detected_unix=time.time(), event_id=w["event_id"],
            quality=dict(conditions=self.cond.around(b["t_bottom"])))
        win = w.pop("_win")
        raw_sigma = w.pop("_raw_sigma")
        row, t_idx, wd = w.pop("_track")
        peak = w.pop("_peak", None)
        e, a = self.band.edges_mhz(), self.searcher.a_hp
        c.flux = self._flux(channel_contributions(win, e, row, t_idx, wd, a_hp=a),
                            raw_sigma, b["tier"], b["width_s"], c.el, c.az, c.snr,
                            contribution_noise(win, e, row, t_idx, wd, a_hp=a), peak)
        if peak is not None:
            c.quality["peak"] = peak
        plan = fdmt_plan(self.band.edges_mhz(), self.searcher.D)
        products = dict(dynspec=win.astype(np.float32), dynspec_t=w.pop("_t_win"),
                        dmt=fdmt(win[:, :, None], plan)[:, :, 0].astype(np.float32),
                        dmt_dm=np.arange(self.searcher.D + 1) * b["dm_step"],
                        freqs_mhz=self.band.centres_mhz(), sigma=w.pop("_sigma"),
                        raw_sigma=raw_sigma, tier_images=np.int32(2 ** b["tier"]),
                        image_s=np.float64(self.dt),
                        watch=np.array([w["ra"], w["dec"], w["dm"], w["t_inf"], w["dm_err"],
                                        w["t_tol"]]))
        stored = self.store.add(c, products, self.radius, dedupe=False)
        w.update(state="measured", result=dict(cand_id=c.id, snr=c.snr, dm=c.dm,
                                               t_inf=c.t_inf, t_bottom=c.t_bottom, ra=ra, dec=dec,
                                               el=c.el, conditions=c.quality["conditions"],
                                               fluence_jy_ms=(c.flux or {}).get("fluence_jy_ms"),
                                               fluence_err_jy_ms=(c.flux or {}).get(
                                                   "fluence_err_jy_ms")))
        print(f"skymon: followup {c.id} for {w['event_id']} S/N {c.snr:.1f} DM {c.dm:.1f}",
              flush=True)
        return stored

    def _save_context(self, w):
        """
        For a watch that could not be measured, keep what this band still has:
        the accumulated spectrum of the position around the predicted arrival.
        """
        if w.get("context_done"):
            return
        w["context_done"] = True
        tb = predicted_bottom(w, self.band.f_lo)
        with self.grid_lock:
            s = self.grid.nearest_slot(w["ra"], w["dec"])
            hpix = int(self.grid.hpix_of[s]) if s >= 0 else -1
        if s < 0:
            return
        t, d = self.accum.pixel(s, since=tb - 1800)
        keep = t <= tb + 1800
        if not keep.any():
            return
        el, az = altaz(w["ra"], w["dec"], tb)
        cid = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime(w['t_inf']))}_" \
              f"{self.band.f_centre:.0f}MHz_ctx_{uuid.uuid4().hex[:6]}"
        c = Candidate(
            id=cid, cls="context", snr=0.0, dm=w["dm"], dm_step=w["dm_err"], width_s=0.0,
            t_inf=w["t_inf"], t_bottom=tb, t_top=tb, utc_inf=utc(w["t_inf"]), hpix=hpix,
            nside=self.grid.nside, ra=float(self.grid.ra[hpix]), dec=float(self.grid.dec[hpix]),
            el=float(el), az=float(az), image_x=float("nan"), image_y=float("nan"), tier=-1,
            tier_dt=self.accum.interval, n_pix=1, extent_deg=0.0, f_lo_mhz=self.band.f_lo,
            f_hi_mhz=self.band.f_hi, host=self.cfg.host, detected_unix=time.time(),
            event_id=w["event_id"], quality=dict(note=w.get("reason") or "not measured",
                                                 conditions=self.cond.around(tb)))
        self.store.add(c, dict(accum=d[keep], accum_t=t[keep], freqs_mhz=self.band.centres_mhz(),
                               watch=np.array([w["ra"], w["dec"], w["dm"], w["t_inf"],
                                               w["dm_err"], w["t_tol"]])),
                       self.radius, dedupe=False)
        w["context_id"] = cid

    def _expire_watches(self, unix):
        """Give up on watches whose time has long passed, or that no tier can search"""
        for w in self.watches.list_live():
            if w["state"] == "expired" and not w.get("context_done") and \
                    unix > predicted_bottom(w, self.band.f_lo) + 30:
                self._save_context(w)
        for w in self.watches.pending():
            if w["dm"] > self.searcher.tiers[-1].dm_max:
                w.update(state="expired", reason="DM beyond the search range")
                continue
            tier = self.searcher.tier_for_dm(w["dm"])
            tb = predicted_bottom(w, self.band.f_lo)
            if unix > tb + w["t_tol"] + 3 * tier.block * tier.dt + 120:
                if w["best"] is not None:
                    self._finish_watch(w)
                elif w["created"] > tb + w["t_tol"]:
                    w.update(state="expired", reason="the burst passed this band before the "
                                                     "watch arrived; covered by the blind search")
                else:
                    w.update(state="expired", reason=w.get("reason") or "not measured")

    # ------------------------------------------------------------ the local database
    def sync_db(self):
        """
        Exchange the mailbox with the node's database: take new watches and
        injection requests, write back watch results, replay the spool, and
        record the status and the queued conditions samples. Runs in its own
        thread (start_db_sync), never in the processing thread.
        """
        db = self.db
        if db is None:
            return
        news = db.new_watches({w["id"] for w in self.watches.list()})
        for req in news or []:
            try:
                self.watches.add(req)
            except ValueError as e:
                db.update_watch(dict(id=req["id"], state="expired", reason=f"bad request: {e}"))
        for w in self.watches.list_live():
            sig = (w["state"], w.get("context_id"))
            if w.get("_synced") != sig and (w["state"] != "pending" or w.get("context_id")):
                out = {k: v for k, v in w.items() if not k.startswith("_")}
                if db.update_watch(out):
                    w["_synced"] = sig
        injs = db.new_injections({b.id for b in self.injector.active + self.injector.waiting})
        for spec in injs or []:
            try:
                self.injector.add_external(spec)
                db.set_injection(spec["id"], "accepted")
            except ValueError:
                db.set_injection(spec["id"], "rejected")
        if db.spooled():
            db.replay()
        self.cond.write(db)
        while self.calfeed.unwritten:
            r = self.calfeed.unwritten[0]
            if not db.put_calibration("record", r.get("t_solved_unix") or time.time(),
                                      r.get("gain_version"), r):
                break
            self.calfeed.unwritten.pop(0)
        while self.flux_unwritten:
            m = self.flux_unwritten[0]
            if not db.put_calibration("measurement", m["t"], m.get("gain_version"), m):
                break
            self.flux_unwritten.popleft()
        st = self.status()
        st.pop("injections", None)
        db.put_status("monitor", st)

    def start_db_sync(self, stop: threading.Event, every_s: float = 2.0):
        def loop():
            while not stop.is_set():
                try:
                    self.sync_db()
                except Exception as e:  # never let the mailbox stop the search
                    print(f"skymon: database sync failed: {e!r}", flush=True)
                stop.wait(every_s)
        th = threading.Thread(target=loop, daemon=True)
        th.start()
        return th

    def watch_list(self, event_id=None):
        out = []
        for w in self.watches.list(event_id):
            out.append({k: v for k, v in w.items() if not k.startswith("_")})
        return out

    # ------------------------------------------------------------ queries
    def status(self):
        """Counters and configuration; reads without locks so it never waits for a search"""
        m = dict(self.metrics)
        st = dict(host=self.cfg.host, started=self.started, now=time.time(),
                  images=int(m.get("images", 0)),
                  missing_images=int(m.get("missing_images", 0)),
                  late_images=int(m.get("late_images", 0)),
                  tracked=int(m.get("tracked", 0)),
                  last_image_unix=self.last_image_unix,
                  lag_s=(time.time() - self.last_image_unix) if self.last_image_unix else None,
                  cpu_fraction=(m.get("process_s", 0) / (m.get("images", 1) * self.dt))
                  if self.band else None,
                  candidates=self.store.counts, injections=self.injector.summary(),
                  database=None if self.db is None else dict(
                      ready=self.db.ready, errors=self.db.errors, last_error=self.db.last_error,
                      spooled=self.db.spooled()),
                  write_queue=self.store.q.qsize(),
                  conditions=self.cond.status(),
                  calibration=dict(self.calibration_status(), records=self.calfeed.status(),
                                   flux=None if self.flux_meas is None else dict(
                                       t=self.flux_meas["t"],
                                       gain_version=self.flux_meas["gain_version"],
                                       n_used=self.flux_meas["n_used"],
                                       calibrators=[dict(name=c["name"], el_deg=c["el_deg"],
                                                         excess_snr=c["excess_snr"])
                                                    for c in self.flux_meas["calibrators"]])))
        if self.band:
            st.update(band=dict(f_lo=self.band.f_lo, f_hi=self.band.f_hi,
                                nchan=self.band.nchan, chan0=self.band.chan0,
                                nbin=self.band.nbin, dt=self.dt),
                      nside=self.grid.nside, pix_deg=self.grid.pix_deg, beam_deg=self.beam,
                      tiers=self.searcher.describe(), el_min=self.cfg.el_min,
                      threshold=self.cfg.threshold,
                      memory_gib=(self.searcher.memory_bytes() +
                                  self.accum.memory_bytes()) / 2 ** 30)
            rfi = self.searcher.rfi_status()
            rfi["masked_mhz"] = [round(float(f), 4) for f in
                                 self.band.centres_mhz()[rfi["masked_channels"]]]
            st["rfi_flagging"] = rfi
        return st

    def pixel(self, ra=None, dec=None, hpix=None, x=None, y=None, since=None, fine=False):
        """Spectrogram of one tracked pixel: dict with times, freqs and data [T][C]"""
        import healpy as hp
        with self.grid_lock:
            if self.grid is None:
                return None
            if x is not None and y is not None:
                ra, dec = self.wcs.all_pix2world(float(x), float(y), 1, ra_dec_order=True)
                ra, dec = float(ra), float(dec)
            if hpix is None:
                hpix = hp.ang2pix(self.grid.nside, math.radians(90 - dec), math.radians(ra))
            hpix = int(hpix)
            slot = int(self.grid.slot_of[hpix])
            out = dict(hpix=hpix, nside=self.grid.nside, ra=float(self.grid.ra[hpix]),
                       dec=float(self.grid.dec[hpix]), tracked=slot >= 0,
                       freqs_mhz=self.band.centres_mhz().tolist())
            if slot < 0:
                out.update(times=[], data=[])
                return out
            if fine:
                tier = self.searcher.tiers[0]
                n = tier.pos
                win = tier.history[:, :n, slot].T.copy()
                first = tier.pushed - n
                t = self.searcher.tier_time(tier, first + np.arange(n))
                out.update(kind="fine", units="normalised", times=t, data=win)
            else:
                t, d = self.accum.pixel(slot, since)
                out.update(kind="accumulation", units="raw", interval_s=self.accum.interval,
                           times=t, data=d)
            el, az = altaz(out["ra"], out["dec"], self.last_image_unix or time.time())
            out.update(el=float(el), az=float(az))
            return out

    def conditions(self, since=None):
        """The conditions series (oldest first) with the thresholds and the latest summary"""
        st = self.cond.status()
        return dict(host=self.cfg.host, thresholds=st["thresholds"], summary=st["summary"],
                    samples=self.cond.samples(since))

    def sky(self):
        with self.grid_lock:
            if self.grid is None:
                return dict(nside=None, hpix=[])
            act = np.nonzero(self.grid.hpix_of >= 0)[0]
            sig = np.sqrt(np.median(self.searcher.var[:, act], axis=0))
            return dict(nside=self.grid.nside, hpix=self.grid.hpix_of[act].tolist(),
                        age_s=(self.searcher.age[act] * self.dt).tolist(),
                        sigma=sig.tolist())
