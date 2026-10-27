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
from .candidates import (Candidate, CandidateStore, FixedDirections, cluster_hits,
                         elevation_radius, separation_deg, unit_vectors, utc)
from .dedisp import K_DM
from .geometry import Band, SkyGrid, altaz, auto_nside, beam_fwhm_deg, image_wcs
from .inject import LiveInjector
from .known import KnownSources
from .search import Searcher
from . import burstfit
from .flux import (CalibrationFeed, beam_sigma_pix, beam_table, calibrators_from_peaks, candidate_flux,
                   measure_calibrators,
                   peak_fit)
from .quality import candidate_quality, channel_contributions, contribution_noise
from .watch import Watches, measure, predicted_bottom


@dataclass
class Config:
    out_dir: str = "skymon_out"  # on local disk: candidates, spool, logs (never the archive)
    db_dsn: str | None = None  # the node's local PostgreSQL (localdb.py); None: files only
    nside: int = 0  # 0: from the beam at the band centre
    el_min: float = 10.0
    # candidates are reported only this many degrees above el_min: the pixels
    # between are tracked and searched for their history, which a pixel that
    # has just risen above the limit lacks. On live data (2026-10-23/24) 73% of
    # the candidates lay within 5 deg of el_min, whether it was 14 or 20 deg
    el_guard: float = 0.0
    dm_max: float = 3000.0
    # S/N threshold of the reference search (boxcars 1, 2, 4, 8 in every
    # tier); with far_matched the threshold applied is raised for the extra
    # trials of wider boxcars, at the same Gaussian false-alarm rate
    threshold: float = 8.0
    far_matched: bool = True
    # boxcar widths in tier samples: one list for every tier, or a spec per
    # tier range such as "0-4:1,2,4,8,16,32,64;5-:1,2,4,8" (search.tier_spec)
    # (the wide-pulse search: boxcars to 128 samples in tiers 0-4, which hold
    # DM up to 1450-2900 at 57-76 MHz, and to 8 above; docs/skymon/DESIGN.md
    # section 3.3)
    widths: tuple | str = "0-4:1,2,4,8,16,32,64,128;5-:1,2,4,8"
    block0: int = 256
    # delay rows of each tier's FDMT as a multiple of the channels: with 2 each
    # tier reaches twice the DM at its time resolution, so a burst is searched
    # at a finer time resolution against its smearing (+18-33% S/N above
    # tier 0) for about 1.45 times the search time with block0 512
    delay_factor: float = 1.0
    nthreads: int = 8
    regrid_interval_s: float = 10.0
    max_gap_s: float = 60.0
    # images held after a change of the calibration gain table: their mean
    # becomes each pixel's baseline before they are searched (0: off)
    rebase_images: int = 24
    # the burst model fitted to candidates (burstfit.py): at most this many a minute, with this
    # many evaluations of the model each (0 a minute: none), in this many worker processes
    # beside the search (0: in the search's thread, as a test reference)
    burst_fits_per_min: int = 6
    burst_fit_nfev: int = 200
    burst_fit_workers: int = 2
    warmup_images: int = 64
    norm_tau_s: float = 120.0
    var_tau_s: float = 300.0
    # tier high-pass time constant in tier samples (a number or a per-tier
    # spec), at least hp_factor times the tier's widest boxcar
    hp_tau: float | str = 32.0
    hp_factor: float = 4.0
    # each tier's history (the span of the trend fit) holds at least
    # detrend_len samples and detrend_factor times its widest boxcar, besides
    # the max_delay + block samples of the transform
    detrend_len: int | str = 0
    detrend_factor: float = 0.0
    # "mean" (default) or "linear": a straight line fitted over the window
    # takes up to 4 w / W of a pulse of w samples at the window's end
    detrend: str = "mean"
    # a boxcar of w samples is evaluated every w / boxcar_stride_div samples
    # (a power of two; 0: every sample), Searcher stride_div
    boxcar_stride_div: int = 8
    # a candidate from a boxcar wider than 8 samples within this many degrees
    # of el_min is classified as RFI (data kept): rising pixels at the horizon
    wide_el_margin: float = 3.0
    noise_every: int = 1  # the search noise is measured in one block of this many
    # the tiers' histories in half precision ("float16"): half the memory of
    # the search state and of its memory traffic; the sums are formed in
    # single precision (search.Searcher history_dtype)
    history_dtype: str = "float16"
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
    # live injections are scattered (tau at 600 MHz log-uniform over
    # inject_tau600_ms, scaled as nu^-4), except a fraction left unscattered
    inject_scatter: bool = True
    inject_tau600_ms: tuple = (0.31, 10.0)
    inject_unscattered_fraction: float = 0.2
    store_rfi: bool = True
    # single pulses of ordinary pulsars: stored as candidates (True), or only tallied
    # per pulsar in the status (False); giant pulses (the Crab's) are always stored
    store_pulsar_pulses: bool = False
    # records kept per search block for groups that are certainly RFI (zero DM
    # or much of the sky at once), brightest first; the rest are only counted
    # (conditions RFI rate, status rfi_unrecorded). An interference storm
    # otherwise costs a record per group and stalls the search
    rfi_records_max: int = 20
    # a group at most this fraction of the amplitude of a group elsewhere at
    # the same time is that burst seen through a sidelobe of the array beam
    # (the strongest single-channel sidelobe of LWA-SV is about 2.3 percent
    # at 76 MHz): RFI, not certain, its data kept. A very bright transient
    # (a solar burst, a flash of interference) puts about 0.4 percent of
    # itself on every pixel, enough to cross the threshold on a few
    sidelobe_max: float = 0.03
    # a candidate at an azimuth and elevation where at least this many earlier
    # detections (in the last fixed_dir_hours) were made while the sky was
    # elsewhere is tied to the ground or the instrument: RFI, data kept
    # write the pixel table with which an FX imager regrids on the GPU (shm.SkyTableWriter)
    pixel_table: bool = True
    fixed_dir_repeats: int = 2
    fixed_dir_hours: float = 6.0
    # a candidate within this many seconds of a solar burst (a burst at the
    # Sun, in this band) is not reported as a candidate but as RFI pending
    # confirmation (data kept): the combiner opens an event for it, which only
    # a detection in another band confirms. Type III bursts drift like a
    # dispersed pulse and are seen through the sidelobes over the whole sky;
    # 68% of the daytime candidates of 2026-10-23/24 lay within 30 s of one.
    # 0 disables
    solar_window_s: float = 30.0
    # the Marchenko-Pastur cut (svdclean.py): remove the components shared by
    # many pixels before the search, in blocks of at most svd_window images
    # (ended at every update of the tracked pixels), with the components of
    # svd_subbands groups of channels. The search runs that many images late
    svd_clean: bool = False
    svd_subbands: int = 8
    svd_window: int = 256
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
    # widths (s) at which the measured search noise is recorded against
    # white noise in the conditions samples (Searcher.noise_at_widths)
    cond_noise_widths_s: tuple = (1.0, 7.0, 20.0)
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
                                     cfg.inject_dm, rng=self.rng, scatter=cfg.inject_scatter,
                                     tau600_ms=cfg.inject_tau600_ms,
                                     unscattered_fraction=cfg.inject_unscattered_fraction)
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
        self._fit_times = collections.deque()
        # the peaks of the sources the imager peels from the pixels (shm.PEEL_NAMES), per image
        self._peel_peaks = collections.deque(maxlen=4096)
        self._solar_times = collections.deque(maxlen=20000)  # t_bottom of recent solar bursts
        self._term, self._term_table = None, None  # the display's peeled-source term (_display_term)
        # burst fits running in the worker processes: (future, candidate, tier level, submitted)
        self._fit_pool = None
        self._fits: list[tuple] = []
        # candidates whose data are still being extended past the burst (_extend_tails)
        self._tails: list[dict] = []
        # ordinary pulsars' single pulses, tallied rather than stored (_tally_pulse)
        self.pulsar_pulses: dict[str, collections.deque] = {}
        # set by the ring reader (run_ring): the pixel table for the imager's GPU regridding
        self.table_writer = None
        self.table_version = 0
        self.fixed_dirs = FixedDirections(cfg.fixed_dir_hours)
        self.lag_hist = collections.deque(maxlen=600)
        self.last_image_unix = None
        self.config_key = None

    # ------------------------------------------------------------ setup
    def _setup(self, meta):
        cfg = self.cfg
        self._held = None  # images held after a gain change (_push_or_hold)
        self._gv_last = None
        self.band = Band(meta.chan0, meta.nchan, meta.nbin)
        self.dt = meta.img_len_ms / 1e3
        self.nseqs = meta.nseqs
        self.seq0 = meta.seq_start
        self.unix0 = meta.unix
        nside = cfg.nside or auto_nside(self.band.f_centre, meta.grid_res)
        self.grid = SkyGrid(nside, cfg.el_min, meta.grid, meta.grid_res, meta.phase_mode)
        self.grid.el_report = cfg.el_min + cfg.el_guard  # injections go above it
        self.beam_tab = None  # the primary beam of this band (flux.py)
        if cfg.beam_model != "record":
            try:
                e = self.band.edges_mhz()
                self.beam_tab = beam_table([e[0], self.band.f_centre, e[-1]], cfg.beam_model)
            except Exception as ex:  # no lsl: the calibration record's beam, if any
                print(f"skymon: primary beam ({cfg.beam_model}) unavailable: {ex!r}", flush=True)
        self.searcher = Searcher(self.band.edges_mhz(), self.dt, self.grid.capacity,
                                 dm_max=cfg.dm_max, widths=cfg.widths, block0=cfg.block0,
                                 max_delay=max(1, int(round(cfg.delay_factor * self.band.nchan))),
                                 threshold=cfg.threshold, nthreads=cfg.nthreads,
                                 warmup_images=cfg.warmup_images, norm_tau_s=cfg.norm_tau_s,
                                 var_tau_s=cfg.var_tau_s, hp_tau=cfg.hp_tau, clip=cfg.clip,
                                 unix0=self.unix0, rng=self.rng,
                                 chan_flag_sigma=cfg.chan_flag_sigma, flag_tau_s=cfg.flag_tau_s,
                                 mask_on=cfg.mask_on, mask_off=cfg.mask_off,
                                 clip_nbr_min=cfg.clip_sky_neighbours,
                                 clip_nbr_frac=cfg.clip_sky_fraction, hp_factor=cfg.hp_factor,
                                 detrend_len=cfg.detrend_len, detrend_factor=cfg.detrend_factor,
                                 detrend=cfg.detrend, far_matched=cfg.far_matched,
                                 stride_div=cfg.boxcar_stride_div,
                                 noise_every=cfg.noise_every,
                                 history_dtype=cfg.history_dtype)
        self.accum = Accumulator(self.band.nchan, self.grid.capacity, self.dt,
                                 cfg.accum_interval_s, cfg.accum_hours)
        self.raw = np.zeros((self.band.nchan, self.grid.capacity), np.float32)
        self.svd = None
        self._svd_buf, self._svd_out = [], []
        if cfg.svd_clean:
            from .svdclean import SvdCleaner
            self.svd = SvdCleaner(self.band.nchan, cfg.svd_subbands, cfg.svd_window,
                                  nthreads=cfg.nthreads)
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
                  f"{t['dm_max']:.1f} step {t['dm_step']:.2f}, latency {t['latency_s']:.0f} s, "
                  f"boxcars {','.join(map(str, t['widths']))} (to {t['max_width_s']:.1f} s), "
                  f"high-pass {t['hp_tau']:g}, history {t['history']}", flush=True)
        mem = self.searcher.memory_bytes() + self.accum.memory_bytes()
        print(f"skymon: S/N threshold {self.searcher.threshold:.3f} "
              f"(reference {self.searcher.threshold_ref:g}); state memory "
              f"{mem / 2 ** 30:.1f} GiB", flush=True)

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
        if self.table_writer is not None:
            self._write_table(unix)
        self.metrics["tracked"] = int(self.grid.active().sum())

    def _write_table(self, unix):
        """
        The pixel table for the imager's GPU regridding (shm.SkyTableWriter).
        The imager works ahead of the monitor by the monitor's lag (tens of
        seconds while it starts), so the table spans 120 s, positions at unix
        and at unix + 120 s, between which the imager interpolates and up to
        half a span beyond; the sky's motion is linear over it to 0.01 pixel
        (the curvature of a 0.5 deg arc on an image of 0.53 deg pixels). The
        table is rewritten at every update of the pixels.
        """
        from .geometry import image_wcs

        g = self.grid
        t_b = unix + 120.0
        xy_b = np.full_like(g.xy_a, np.nan)
        act = np.nonzero(np.isfinite(g.xy_a[:, 0]))[0]
        if len(act):
            wcs_b = image_wcs(t_b, g.grid_size, g.grid_res, g.phase_mode)
            x, y = wcs_b.all_world2pix(g.ra[g.hpix_of[act]], g.dec[g.hpix_of[act]], 1, ra_dec_order=True)
            xy_b[act, 0], xy_b[act, 1] = x, y
        try:
            self.table_version = self.table_writer.write(g.grid_size, g.grid_res, unix, t_b, g.xy_a, xy_b)
        except OSError as e:
            print(f"skymon: pixel table not written: {e}", flush=True)

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
                out += self._push_or_hold(None, self.last_index + 1)
            out += self._push_or_hold((meta, image), n)
            self.last_image_unix = meta.unix
            return out

    def _push_or_hold(self, item, n):
        """
        Search the image, or hold it after a change of the calibration.

        Every gain table changes the images a little, and differently on each
        pixel (sources and sidelobes move against the sky median, which the
        normalisation removes): a step that each pixel's running mean, with
        its time constant of minutes, would take for a broadband pulse.
        After a change the next rebase_images images are held; their mean
        becomes the baseline of every pixel and channel (Searcher.rebase),
        and then they are searched in order. Nothing is lost; the search runs
        that many images late at each change.
        """
        K = self.cfg.rebase_images
        if K > 0 and item is not None:
            v = int(item[0].gain_version)
            last = getattr(self, "_gv_last", None)
            self._gv_last = v
            if last is not None and v != last:
                # images before the change are searched before the new baselines
                out = self._svd_flush()
                out += self._release() if getattr(self, "_held", None) is not None else []
                self._held = []
                self._held_sum = np.zeros_like(self.raw, dtype=np.float64)
                self._held_n = np.zeros(self.raw.shape[1], np.int32)
                self._held_table = getattr(self, "table_version", None)
                self.metrics["rebased"] = self.metrics.get("rebased", 0) + 1
                return out + self._hold(item, n)
        if getattr(self, "_held", None) is not None:
            return self._hold(item, n)
        return self._push(item, n)

    def _hold(self, item, n):
        prep = self._prepare(item, n)
        raw = None
        if prep["item"] is not None:
            raw = self.raw.copy()
            self._held_sum += raw
            self._held_n += 1
        self._held.append((prep, raw))
        return self._release() if len(self._held) >= self.cfg.rebase_images else []

    def _release(self):
        """New baselines from the held images, then search them"""
        held, self._held = self._held, None
        if getattr(self, "table_version", None) == self._held_table and self._held_n.max() > 0:
            ok = self._held_n > 0
            mean = (self._held_sum[:, ok] / self._held_n[ok]).astype(np.float32)
            first = next(r for _, r in held if r is not None)
            self.searcher.rebase(np.nonzero(ok)[0], mean, first[:, ok])
        self._held_sum = None
        out = []
        for prep, raw in held:
            if raw is not None:
                np.copyto(self.raw, raw)
            out += self._searchq(prep)
        return out

    def _push(self, item, n):
        return self._searchq(self._prepare(item, n))

    def _searchq(self, prep):
        """_search, through the block buffer of the Marchenko-Pastur cut when it is on.
        prep's values are in self.raw. Returns the results of every image searched now"""
        out, self._svd_out = self._svd_out, []
        if self.svd is None:
            return out + self._search(prep)
        self._svd_buf.append((prep, self.raw.copy() if prep["item"] is not None else None))
        if len(self._svd_buf) >= self.svd.window:
            out += self._svd_flush()
        return out

    def _svd_flush(self):
        """Clean the buffered block with its own components (after the injections, so the
        measured efficiency includes the cleaning), then search it in order"""
        buf, self._svd_buf = self._svd_buf, []
        if not buf:
            return []
        idx = [i for i, (_, r) in enumerate(buf) if r is not None]
        if len(idx) >= 32:  # too few images for the components: searched as they are
            blk = np.stack([buf[i][1] for i in idx])
            self.svd.clean(blk, self.searcher.mu, self.searcher.var)
            for j, i in enumerate(idx):
                buf[i] = (buf[i][0], blk[j])
        out = []
        for prep, raw in buf:
            if raw is not None:
                np.copyto(self.raw, raw)
            out += self._search(prep)
        return out

    def _prepare(self, item, n):
        """Everything of one image before the search: the pixels' values in self.raw"""
        t0 = time.perf_counter()
        unix = self.unix0 + n * self.dt
        if self.cond.due(unix):
            self._sample_conditions()
        self.cond.note_image(item is None)
        if self.grid_next is None or unix >= self.grid_next:
            # a cleaning block never spans two pixel sets: search what is buffered first
            self._svd_out += self._svd_flush()
            self._update_grid(unix)
        else:
            self.grid.taps(unix, self.cfg.nthreads)
        self.last_index = n
        if item is not None and item[0].layout == 1 and item[1].shape != self.raw.shape:
            # spectra for another pixel set (the table changed size): an image lost
            self.metrics["missing_images"] += 1
            self.metrics["pixel_mismatch"] += 1
            item = None
        if item is not None:
            meta, image = item
            self._note_gains(meta.gain_version, unix)
            self.injector.schedule(unix, self.grid, self.searcher,
                                   self.cfg.warmup_images + 4 * self.cfg.block0)
            self.injector.prepare(unix, self.grid, self.searcher)
            if meta.layout == 1:
                # regridded on the GPU by the imager (FX imaging); injections
                # go into the spectra
                np.copyto(self.raw, image)
                # only from calibrated images: with the nominal phases the fitted fluxes
                # have arbitrary phases, and their real parts are no measure of the flux
                if meta.peel_peaks is not None and int(meta.gain_version) > 0:
                    from .shm import PEEL_NAMES
                    self._peel_peaks.append((unix, {nm: meta.peel_peaks[i] for i, nm in enumerate(PEEL_NAMES)
                                                    if meta.peel_mask >> i & 1}))
                self.metrics["gpu_regridded"] += 1
                if self.injector.active:
                    self.injector.apply_pixels(self.raw, self.grid.xy, unix, self.dt,
                                               self.band.edges_mhz(), self.band.centres_mhz(),
                                               self.wcs, self.cfg_pix_deg(meta))
            else:
                if self.injector.active:
                    self.injector.apply(image, unix, self.dt, self.band.edges_mhz(),
                                        self.band.centres_mhz(), self.wcs, self.cfg_pix_deg(meta))
                kernels.regrid(image, self.grid.idx, self.grid.w, self.raw, self.cfg.nthreads)
        return dict(item=item, unix=unix, dt_s=time.perf_counter() - t0)

    def _near_solar_burst(self, t: float) -> bool:
        """A solar burst in this band within solar_window_s of t (recent bursts only)"""
        w = self.cfg.solar_window_s
        if w <= 0:
            return False
        while self._solar_times and self._solar_times[0] < t - 3600.0:
            self._solar_times.popleft()
        return any(abs(t - s) <= w for s in self._solar_times)

    def _display_term(self, meta):
        """The peeled sources at each pixel for the display accumulation. The imager
        sends them with every few images; the last is reused in between, and dropped
        when peeling stops or the pixel table changes (slots change their pixels)."""
        if not getattr(meta, "peel_mask", 0) or getattr(meta, "table_version", 0) != self._term_table:
            self._term = None
        term = getattr(meta, "peel_term", None)
        if term is not None and term.shape == self.accum.sum.shape:
            self._term, self._term_table = term, meta.table_version
        return self._term

    def _search(self, prep):
        """The search of one prepared image (_prepare; self.raw holds its pixels)"""
        t0 = time.perf_counter()
        unix = prep["unix"]
        if prep["item"] is None:
            results = self.searcher.push(None)
        else:
            self.accum.begin(unix)
            results = self.searcher.push(self.raw, self.accum.sum)
            # the display accumulation keeps the unpeeled sky: the search runs on the
            # peeled spectra, the sky map, spectrograms and calibrators see the sources
            term = self._display_term(prep["item"][0])
            if term is not None:
                np.add(self.accum.sum, term, out=self.accum.sum)
            self.accum.end_image(unix + self.dt)
        found = []
        for res in results:
            self.cond.note_block(res)
            self._extend_tails(res)
            found += self._candidates(res)
            found += self._measure_watches(res)
        self._expire_watches(unix)
        self._collect_fits()
        self.searcher.finish_block()
        self.injector.expire(unix)
        self.metrics["images"] += 1
        self.metrics["process_s"] += time.perf_counter() - t0 + prep["dt_s"]
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
            # the calibrators the imager peeled are not in the pixels: their fitted peaks
            # stand in for the measurement (the others are measured in the pixels)
            m = None
            if self._peel_peaks and unix - self._peel_peaks[-1][0] < 2 * self.cfg.flux_every_s:
                m = calibrators_from_peaks(list(self._peel_peaks), rec, self.band.centres_mhz(), unix,
                                           beam_tab=getattr(self, "beam_tab", None))
            # the calibrators in the pixels (those not peeled), measured in any case
            why = []
            with self.grid_lock:
                mp = measure_calibrators(self.grid, self.accum, self.beam, rec,
                                         self.band.centres_mhz(), unix,
                                         beam_tab=getattr(self, "beam_tab", None), why=why)
            self.flux_meas_pixels = mp
            if mp is None:
                print(f"skymon: no calibrator measured in the pixels: {'; '.join(why) or 'none in the record'}",
                      flush=True)
            if m is None:
                m = mp
            if m is not None:
                self.flux_meas = m
                self.flux_unwritten.append(m)
        except Exception as e:  # flux is for review; it must never stop the search
            print(f"skymon: flux calibration failed: {e!r}", flush=True)

    def _peak(self, window, slot, row, t_idx, width, a_hp):
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
                vals = np.array([float(np.sum(channel_contributions(
                    np.asarray(window[:, :, q], np.float32), e, row, t_idx, width, a_hp=a_hp)))
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

    def _noise_record(self, rec_t):
        """
        The realised noise floor for the conditions sample: the measured
        search noise relative to white noise at widths of about 1, 7 and 20 s
        (Searcher.noise_at_widths), and the elevations of Cyg A and Cas A,
        whose scintillation and beam passage raise the noise at these widths.
        """
        out = {}
        try:
            out["search_noise"] = self.searcher.noise_at_widths(self.cfg.cond_noise_widths_s)
        except Exception as e:  # a record for review must never stop the search
            out["search_noise"] = dict(error=repr(e))
        for key, (ra, dec) in (("el_cyg_a", (299.868, 40.734)), ("el_cas_a", (350.850, 58.815))):
            try:
                el, _ = altaz(ra, dec, rec_t)
                out[key] = round(float(el), 2)
            except Exception:
                out[key] = None
        return out

    def _sample_conditions(self):
        """
        Close a conditions interval (every cond_interval_s of images). The
        sample is only queued for the database, which the database thread
        writes (sync_db), so the search never waits for it.
        """
        s = self.searcher
        rec = self.cond.sample(s.var, s.age, s.live, s.warmup, self.metrics["tracked"],
                               extra=self._noise_record(rec_t=self.cond.next_t))
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
        n_certain = 0
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
        # (all pairs of group heads at once, in row chunks to bound memory)
        # The same pairs give, for each group, the brightest group elsewhere at
        # the same time, in image units (S/N times the pixel's raw noise), for
        # the sidelobe test
        heads = np.array([g[0] for g in groups], int)
        sites = np.zeros(len(groups), int)
        hv = unit_vectors(ra[heads], dec[heads])
        h_lo, h_hi, h_r = t_lo[heads], t_hi[heads], radius[heads]
        h_slot = slots[heads]
        amp = np.array([hits[k].snr for k in heads]) * \
            np.sqrt(np.maximum(self.searcher.var[:, h_slot], 0).mean(0))
        brighter = np.zeros(len(groups))
        brighter_at = np.full(len(groups), -1)
        for a in range(0, len(heads), 1024):
            b = slice(a, a + 1024)
            sep = np.degrees(np.arccos(np.clip(hv[b] @ hv.T, -1, 1)))
            apart = sep > 2 * np.maximum(h_r[b, None], h_r[None, :])
            overlap = (h_lo[None, :] <= h_hi[b, None]) & (h_hi[None, :] >= h_lo[b, None])
            other = np.where(overlap & apart, amp[None, :], 0.0)
            brighter[b] = other.max(1)
            brighter_at[b] = other.argmax(1)
            apart[np.arange(apart.shape[0]), np.arange(a, a + apart.shape[0])] = True
            sites[b] = np.count_nonzero(overlap & apart, axis=1)
        # solar bursts: groups whose brightest hit is at the Sun (known.py's
        # radius for bodies), and flashes over the sky (seen at once by much of
        # it, or at several separate positions) during one of them, which are
        # the burst seen through the sidelobes. They are labelled, kept apart
        # from RFI in the conditions, and capped like certain RFI
        sun_ra, sun_dec = self.known.body_positions(hits[0].t_bottom)["Sun"]
        at_sun = (separation_deg(ra[heads], dec[heads], sun_ra, sun_dec) <=
                  self.known.body_beams * elevation_radius(self.beam, el_hits[heads])) & \
            self.known.body_up("Sun", hits[0].t_bottom)
        solar_time = np.zeros(len(groups), bool)
        for gi in np.nonzero(at_sun)[0]:
            self._solar_times.append(float(hits[groups[gi][0]].t_bottom))
        if at_sun.any():
            solar_time = ((h_lo[:, None] <= h_hi[at_sun][None, :]) &
                          (h_hi[:, None] >= h_lo[at_sun][None, :])).any(1)
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
            el, az = altaz(ra[g[0]], dec[g[0]], b.t_bottom)
            if el < self.cfg.el_min + self.cfg.el_guard:
                # tracked for its history only: below the reporting limit
                self.metrics["below_report_limit"] = self.metrics.get("below_report_limit", 0) + 1
                continue
            # a pixel near the elevation limit brightens quickly through the
            # primary beam, which a wide boxcar at low DM takes for a pulse
            if b.width > tier.narrow_max and \
                    el < self.cfg.el_min + self.cfg.el_guard + self.cfg.wide_el_margin:
                reasons.append("wide pulse near the horizon")
            solar = bool(at_sun[gi])
            sidelobe = not solar and bool(solar_time[gi]) and (
                "much of the sky at once" in reasons or "several separate positions at once" in reasons)
            if not solar and brighter[gi] > 0 and amp[gi] <= self.cfg.sidelobe_max * brighter[gi]:
                j = int(brighter_at[gi])
                src = "a solar burst" if at_sun[j] else "a burst"
                reasons.append(f"sidelobe of {src} {brighter[gi] / max(amp[gi], 1e-30):.0f} times "
                               f"brighter at RA {ra[heads[j]]:.1f} Dec {dec[heads[j]]:.1f}")
                sidelobe = sidelobe or bool(at_sun[j])
            elif sidelobe:
                reasons.append("solar burst at the same time")
            # certain: terrestrial (no dispersion) or seen over much of the sky
            certain = not solar and ("DM near zero" in reasons or "much of the sky at once" in reasons)
            cls = "known" if solar else ("rfi" if reasons else "candidate")
            counted = "solar" if (solar or sidelobe) else cls
            if certain or solar:  # groups come brightest first
                n_certain += 1
                if n_certain > self.cfg.rfi_records_max:
                    self.cond.note_candidate(counted)
                    self.metrics["solar_unrecorded" if counted == "solar" else "rfi_unrecorded"] += 1
                    continue
            if solar:
                # the tests that would have called it RFI, kept as notes
                solar_notes, reasons = reasons, []
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
                self.cond.note_candidate(counted)  # counted in the RFI rate though not stored
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
            if solar:
                c.known_source = dict(kind="solar", name="Sun", burst=True,
                                      sep_deg=round(float(separation_deg(c.ra, c.dec, sun_ra,
                                                                         sun_dec)), 3))
                c.quality["rfi_tests"] = solar_notes
            elif cls != "rfi":
                inj = self.injector.match(c, float(elevation_radius(self.radius, c.el)))
                if inj is not None:
                    c.cls = "injection"
                    c.injection_id = inj.id
                    self.injector.record(inj, c)
            if not solar:
                # a fixed direction above the horizon where detections keep
                # coming while the sky moves on
                if c.cls == "candidate":
                    n_rep = self.fixed_dirs.repeats(c.t_bottom, c.az, c.el, c.ra, c.dec,
                                                    float(elevation_radius(self.radius, c.el)))
                    if n_rep >= self.cfg.fixed_dir_repeats:
                        c.rfi_reasons.append(
                            f"repeats at a fixed direction (az {c.az:.0f}, el {c.el:.0f}): "
                            f"{n_rep} detections there before, with the sky elsewhere")
                        cls = c.cls = "rfi"
                self.fixed_dirs.add(c.t_bottom, c.az, c.el, c.ra, c.dec)
                if c.cls == "candidate" and self._near_solar_burst(c.t_bottom):
                    c.rfi_reasons.append(f"during a solar burst (within {self.cfg.solar_window_s:.0f} s): "
                                         "a detection in another band is needed")
                    c.quality["needs_confirmation"] = "solar burst"
                    cls = c.cls = "rfi"
            if not certain and not solar:
                self._label_known(c)
            ks = c.known_source or {}
            if c.cls == "known" and ks.get("kind") == "pulsar" and not ks.get("giant_pulse") and \
                    not self.cfg.store_pulsar_pulses:
                self._tally_pulse(ks["name"], c)  # counted, not stored
                self.cond.note_candidate("known")
                continue
            c.quality["conditions"] = self.cond.around(c.t_bottom)
            c.quality["calibration"] = self._calibration_around(c)
            fit = products is not None and c.cls in ("candidate", "injection", "known")
            if fit and self.cfg.burst_fit_workers <= 0:
                c.burst_fit = self._burst_fit(c, res.tier, products)
            stored = self.store.add(c, products, self.beam)  # repeats peak within a beam
            if fit and stored is c and self.cfg.burst_fit_workers > 0:
                c.burst_fit = self._submit_fit(c, res.tier, products)
            if stored is c and products is not None:
                self._start_tail(c, products, res, b)
            if stored is not None:
                self.cond.note_candidate("solar" if (solar or sidelobe) else stored.cls)
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

    def _tally_pulse(self, name: str, c: Candidate):
        """
        A single pulse of an ordinary pulsar, kept only as (arrival, S/N, DM) for the
        last 24 h, at most 5000 a pulsar; the same pulse found again by another tier
        or block (within 2 s) is counted once
        """
        q = self.pulsar_pulses.setdefault(name, collections.deque(maxlen=5000))
        if q and abs(q[-1][0] - c.t_inf) < 2.0:
            if c.snr > q[-1][1]:
                q[-1] = (c.t_inf, round(float(c.snr), 2), round(float(c.dm), 2))
            return
        q.append((c.t_inf, round(float(c.snr), 2), round(float(c.dm), 2)))
        while q and q[0][0] < c.t_inf - 86400.0:
            q.popleft()

    def pulsar_tally(self):
        """Per pulsar: the pulses tallied in the last 24 h, as [arrival, S/N, DM]"""
        return {n: [list(p) for p in q] for n, q in list(self.pulsar_pulses.items()) if q}

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

    def _burst_fit(self, c, tier, products):
        """
        The physical parameters of a candidate (burstfit.py): DM, arrival, intrinsic
        width, scattering and spectrum fitted to its dynamic spectrum, and the
        fitted fluences in Jy ms with the calibration of its flux, in the search's
        thread (burst_fit_workers 0; otherwise _submit_fit). At most
        burst_fits_per_min a minute (each takes about 2 s of one core), so a storm
        of candidates cannot hold up the search; the others get a note.
        """
        now = time.time()
        why = self._fit_allowed()
        if why is not None:
            return dict(skipped=why)
        try:
            f = burstfit.fit(products["dynspec"], products["dynspec_t"], self.band.edges_mhz(), c.dm, c.t_inf,
                             c.width_s, tier_images=2 ** tier.level, a_hp=tier.a_hp, dm_step=c.dm_step,
                             max_nfev=self.cfg.burst_fit_nfev)
            if f is None:
                return dict(error="no fit")
            f["physical"] = burstfit.physical(f, c.flux, tier.level, self.band.centres_mhz())
            f["seconds"] = round(time.time() - now, 2)
            return f
        except Exception as e:  # for review: it must never stop the search
            return dict(error=repr(e)[:200])

    def _fit_allowed(self):
        """The rate limit of burst fits: None if one may start now, else the reason it may not"""
        now = time.time()
        q = self._fit_times
        while q and now - q[0] > 60.0:
            q.popleft()
        if len(q) >= self.cfg.burst_fits_per_min:
            return "rate limit"
        if len(self._fits) >= 4 * self.cfg.burst_fit_workers:
            return "fits queued"
        q.append(now)
        return None

    def _submit_fit(self, c, tier, products):
        """
        Start the burst fit of a stored candidate in a worker process (about 2 s of
        one core each), so the search never waits for it. Its result replaces the
        candidate's burst_fit and its record is written again (_collect_fits).
        """
        why = self._fit_allowed()
        if why is not None:
            return dict(skipped=why)
        try:
            if self._fit_pool is None:
                import concurrent.futures
                import multiprocessing
                # spawned, not forked: the monitor holds tens of GB and OpenMP threads
                self._fit_pool = concurrent.futures.ProcessPoolExecutor(
                    self.cfg.burst_fit_workers, mp_context=multiprocessing.get_context("spawn"))
            fut = self._fit_pool.submit(
                burstfit.fit_job, np.array(products["dynspec"], np.float32),
                np.array(products["dynspec_t"], np.float64), self.band.edges_mhz(), c.dm, c.t_inf,
                c.width_s, 2 ** tier.level, tier.a_hp, c.dm_step, self.cfg.burst_fit_nfev)
        except Exception as e:  # for review: it must never stop the search
            return dict(error=repr(e)[:200])
        self._fits.append((fut, c, tier.level, time.time()))
        return dict(pending=True)

    def _collect_fits(self):
        """The finished burst fits: into their candidates, whose records are written again"""
        if not self._fits:
            return
        keep = []
        for fut, c, level, t0 in self._fits:
            if not fut.done():
                keep.append((fut, c, level, t0))
                continue
            try:
                f = fut.result()
                if f is None:
                    f = dict(error="no fit")
                elif "error" not in f:
                    f["physical"] = burstfit.physical(f, c.flux, level, self.band.centres_mhz())
                f["seconds"] = round(time.time() - t0, 2)
            except Exception as e:  # a worker that died: noted, never fatal
                f = dict(error=repr(e)[:200])
            c.burst_fit = f
            self.store.q.put(("json", c, None))
        self._fits = keep

    def close_fits(self):
        """Stop the fit workers (at exit); fits still running are dropped"""
        if self._fit_pool is not None:
            self._fit_pool.shutdown(wait=False, cancel_futures=True)
            self._fit_pool = None

    # ------------------------------------------------------------ data after the burst
    TAIL_MAX_S = 600.0  # a tail not complete by then is written as it is

    def _start_tail(self, c, products, res, hit):
        """
        The window of a detection ends at the block that completed the burst,
        so the burst sits at its right edge. Keep appending the pixel's samples
        of the following blocks of the same tier until as many samples follow
        the burst as its sweep spans (at least 32, at most the transform's
        delay range), then write the data again (_extend_tails), so the burst
        is seen in the middle of its dynamic spectrum and DM-time plane.
        """
        tier = res.tier
        last = res.first_sample + res.window.shape[1] - 1
        until = hit.t_end + min(self.searcher.D, max(32, hit.row + 4 * hit.width))
        if until <= last:
            return
        self._tails.append(dict(c=c, products=dict(products), level=tier.level, slot=hit.slot,
                                hpix=int(self.grid.hpix_of[hit.slot]), last=last, until=until,
                                started=time.time()))

    def _extend_tails(self, res):
        if not self._tails:
            return
        tier = res.tier
        first, W = res.first_sample, res.window.shape[1]
        keep = []
        for tl in self._tails:
            done = time.time() - tl["started"] > self.TAIL_MAX_S or \
                int(self.grid.hpix_of[tl["slot"]]) != tl["hpix"]  # the pixel set: its slot was reused
            if not done and tl["level"] == tier.level:
                a, b = max(tl["last"] + 1, first), min(tl["until"], first + W - 1)
                if b >= a:
                    p = tl["products"]
                    seg = np.asarray(res.window[:, a - first:b - first + 1, tl["slot"]], np.float32)
                    p["dynspec"] = np.concatenate([p["dynspec"], seg], 1)
                    p["dynspec_t"] = np.concatenate([p["dynspec_t"],
                                                     self.searcher.tier_time(tier, np.arange(a, b + 1))])
                    tl["last"] = b
                done = tl["last"] >= tl["until"]
            if not done:
                keep.append(tl)
                continue
            p = tl["products"]
            try:  # the DM-time plane over the longer span
                win = np.ascontiguousarray(p["dynspec"], np.float32)
                p["dmt"] = kernels.fdmt(win[:, :, None], self.searcher.plan, self.searcher.D)[:, :, 0]
            except Exception as e:  # never let a review product stop the search
                print(f"skymon: DM-time plane of {tl['c'].id} not extended: {e!r}", flush=True)
            self.store.q.put(("npz", tl["c"], p))
        self._tails = keep

    def _products(self, res, hit, cand=None):
        """Data around a candidate: its pixel's window, DM-time plane, and accumulation"""
        tier = res.tier
        s = hit.slot
        win = np.ascontiguousarray(res.window[:, :, s], np.float32)  # [C, W]
        # the compiled transform (equal to dedisp.fdmt term by term)
        dmt = kernels.fdmt(win[:, :, None], self.searcher.plan, self.searcher.D)[:, :, 0]
        if cand is not None:
            try:
                cand.quality = candidate_quality(win, dmt, res.sigma, self.band.edges_mhz(),
                                                 hit.row, hit.t_end - res.first_sample,
                                                 hit.width, list(tier.widths))
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
            a = tier.a_hp
            contrib = channel_contributions(win, e, hit.row, t_idx, hit.width, a_hp=a)
            err = contribution_noise(win, e, hit.row, t_idx, hit.width, a_hp=a)
            peak = self._peak(res.window, s, hit.row, t_idx, hit.width, a)
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
        widths = tier.widths
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
            win = res.window[:, :, slots].astype(np.float32)
            m = measure(win, self.searcher.plan, self.searcher.D, tier, res.sigma, widths,
                        (r0, r1), (t0, t1), detrend=self.searcher.detrend_mode,
                        zerodm=tier.level > 0)
            if m is not None and (w["best"] is None or m[0] > w["best"]["snr"]):
                snr, row, t, wi, k = m
                wd = int(widths[wi])
                t_end = first + t
                t_bottom = self.searcher.tier_time(tier, t_end - (wd - 1) / 2.0 + 0.5)
                dm = row * dpd
                w["best"] = dict(snr=snr, dm=dm, row=row, t_bottom=t_bottom, width_s=wd * tier.dt,
                                 slot=slots[k], tier=tier.level, tier_dt=tier.dt, dm_step=dpd,
                                 t_inf=t_bottom - K_DM * dm / self.band.f_lo ** 2)
                w["_win"] = np.ascontiguousarray(res.window[:, :, slots[k]], np.float32)
                w["_raw_sigma"] = np.sqrt(np.maximum(self.searcher.var[:, slots[k]], 0)).astype(
                    np.float32)
                w["_track"] = (row, t, wd)
                w["_peak"] = self._peak(res.window, slots[k], row, t, wd, tier.a_hp)
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
        e, a = self.band.edges_mhz(), self.searcher.tiers[b["tier"]].a_hp
        c.flux = self._flux(channel_contributions(win, e, row, t_idx, wd, a_hp=a),
                            raw_sigma, b["tier"], b["width_s"], c.el, c.az, c.snr,
                            contribution_noise(win, e, row, t_idx, wd, a_hp=a), peak)
        if peak is not None:
            c.quality["peak"] = peak
        dmt = kernels.fdmt(np.ascontiguousarray(win[:, :, None], np.float32), self.searcher.plan,
                           self.searcher.D)[:, :, 0]
        products = dict(dynspec=win.astype(np.float32), dynspec_t=w.pop("_t_win"),
                        dmt=dmt,
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
                  rfi_unrecorded=int(m.get("rfi_unrecorded", 0)),
                  solar_unrecorded=int(m.get("solar_unrecorded", 0)),
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
                                       method=self.flux_meas.get("method", "pixels"),
                                       k=self.flux_meas.get("k"),
                                       k_pixels=None if getattr(self, "flux_meas_pixels", None) is None
                                       else self.flux_meas_pixels.get("k"),
                                       calibrators=[dict(name=c["name"], el_deg=c["el_deg"],
                                                         excess_snr=c["excess_snr"], k=c.get("k"))
                                                    for c in self.flux_meas["calibrators"]])))
        if self.band:
            st.update(band=dict(f_lo=self.band.f_lo, f_hi=self.band.f_hi,
                                nchan=self.band.nchan, chan0=self.band.chan0,
                                nbin=self.band.nbin, dt=self.dt),
                      nside=self.grid.nside, pix_deg=self.grid.pix_deg, beam_deg=self.beam,
                      tiers=self.searcher.describe(), el_min=self.cfg.el_min,
                      svd_clean=self.svd.status() if self.svd is not None else None,
                      threshold=self.searcher.threshold,
                      threshold_reference=self.searcher.threshold_ref,
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
                win = tier.history[:, :n, slot].T.astype(np.float32)
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
