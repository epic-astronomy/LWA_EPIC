"""
The all-sky monitor of one EPIC imager.

A monitor follows the images of the imager on its own node (shared-memory
ring, shm.py), tracks every HEALPix pixel above the elevation limit, searches
each for dispersed bursts in real time (search.py) and stores what it finds
(candidates.py). It needs nothing from any other node: each worker runs its
own monitor for its own band, and the loss of a worker only removes that band.

Images are processed in time order. A missing image (packet loss, a gap in
the ring) is searched as zeros, so the time axis stays regular; a gap longer
than max_gap_s restarts the search state.
"""
from __future__ import annotations

import collections
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
from .candidates import (Candidate, CandidateStore, cluster_hits, elevation_radius,
                         separation_deg, utc)
from .dedisp import K_DM, fdmt, fdmt_plan
from .geometry import Band, SkyGrid, altaz, auto_nside, beam_fwhm_deg, image_wcs
from .inject import LiveInjector
from .search import Searcher


@dataclass
class Config:
    out_dir: str = "skymon_out"
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
    clip: float = 20.0
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
        self.store = CandidateStore(cfg.out_dir)
        self.injector = LiveInjector(cfg.out_dir, cfg.inject_interval_s, cfg.inject_snr,
                                     cfg.inject_dm, rng=self.rng)
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
        self.searcher = Searcher(self.band.edges_mhz(), self.dt, self.grid.capacity,
                                 dm_max=cfg.dm_max, widths=cfg.widths, block0=cfg.block0,
                                 threshold=cfg.threshold, nthreads=cfg.nthreads,
                                 warmup_images=cfg.warmup_images, norm_tau_s=cfg.norm_tau_s,
                                 var_tau_s=cfg.var_tau_s, hp_tau=cfg.hp_tau, clip=cfg.clip,
                                 unix0=self.unix0, rng=self.rng)
        self.accum = Accumulator(self.band.nchan, self.grid.capacity, self.dt,
                                 cfg.accum_interval_s, cfg.accum_hours)
        self.raw = np.zeros((self.band.nchan, self.grid.capacity), np.float32)
        self.beam = beam_fwhm_deg(self.band.f_centre)
        self.radius = max(2.0 * self.beam, 2.5 * self.grid.pix_deg)
        beam_area = 1.1331 * self.beam ** 2
        self.rfi_max_pix = max(9, int(cfg.rfi_max_beams * beam_area / self.grid.pix_deg ** 2))
        self.last_index = -1
        self.grid_next = None
        self.config_key = (meta.chan0, meta.nchan, meta.nbin, meta.grid, meta.grid_res,
                           meta.nseqs, meta.phase_mode)
        self.wcs_unix = None
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
        self.searcher.set_live(self.grid.active())
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
        if self.grid_next is None or unix >= self.grid_next:
            self._update_grid(unix)
        else:
            self.grid.taps(unix)
        self.last_index = n
        if item is None:
            results = self.searcher.push(None)
        else:
            meta, image = item
            self.injector.schedule(unix, self.grid, self.searcher,
                                   self.cfg.warmup_images + 4 * self.cfg.block0)
            if self.injector.active:
                self.injector.apply(image, unix, self.dt, self.band.edges_mhz(),
                                    self.band.centres_mhz(), self.wcs, self.cfg_pix_deg(meta))
            kernels.regrid(image, self.grid.idx, self.grid.w, self.raw, self.cfg.nthreads)
            self.accum.begin(unix)
            results = self.searcher.push(self.raw, self.accum.sum)
            self.accum.end_image(unix + self.dt)
        found = []
        for res in results:
            found += self._candidates(res)
        self.searcher.finish_block()
        self.injector.expire(unix)
        self.metrics["images"] += 1
        self.metrics["process_s"] += time.perf_counter() - t0
        return found

    def cfg_pix_deg(self, meta):
        return meta.grid_res

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
            cls = "candidate"
            if len(pix) > self.rfi_max_pix / np.clip(np.sin(np.radians(el_hits[g[0]])), 0.25, 1) \
                    or ext > 4 * radius[g[0]] or \
                    flash or b.dm < self.cfg.dm_min:
                cls = "rfi"
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
                members=[[int(hp[i]), round(hits[i].snr, 2)] for i in g[:64]])
            if cls != "rfi":
                inj = self.injector.match(c, float(elevation_radius(self.radius, c.el)))
                if inj is not None:
                    c.cls = "injection"
                    c.injection_id = inj.id
                    self.injector.record(inj, c)
            elif not self.cfg.store_rfi:
                continue
            products = self._products(res, b) if cls != "rfi" else None
            stored = self.store.add(c, products, self.radius)
            if stored is not None:
                found.append(stored)
                print(f"skymon: {stored.cls} {stored.id} S/N {stored.snr:.1f} DM {stored.dm:.1f} "
                      f"RA {stored.ra:.2f} Dec {stored.dec:.2f} tier {tier.level} "
                      f"({len(pix)} px, {n_all} hits)", flush=True)
        return found

    def _products(self, res, hit):
        """Data around a candidate: its pixel's window, DM-time plane, and accumulation"""
        tier = res.tier
        s = hit.slot
        win = np.ascontiguousarray(res.window[:, :, s])  # [C, W]
        plan = fdmt_plan(self.band.edges_mhz(), self.searcher.D)
        dmt = fdmt(win[:, :, None], plan)[:, :, 0]
        t_win = self.searcher.tier_time(tier, res.first_sample + np.arange(win.shape[1]))
        ta, acc = self.accum.pixel(s, since=hit.t_inf - 1800)
        return dict(dynspec=win.astype(np.float32), dynspec_t=t_win, dmt=dmt.astype(np.float32),
                    dmt_dm=np.arange(self.searcher.D + 1) * tier.dm_per_delay,
                    freqs_mhz=self.band.centres_mhz(), accum=acc, accum_t=ta,
                    sigma=res.sigma)

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
                  candidates=self.store.counts, injections=self.injector.summary())
        if self.band:
            st.update(band=dict(f_lo=self.band.f_lo, f_hi=self.band.f_hi,
                                nchan=self.band.nchan, chan0=self.band.chan0,
                                nbin=self.band.nbin, dt=self.dt),
                      nside=self.grid.nside, pix_deg=self.grid.pix_deg, beam_deg=self.beam,
                      tiers=self.searcher.describe(), el_min=self.cfg.el_min,
                      threshold=self.cfg.threshold,
                      memory_gib=(self.searcher.memory_bytes() +
                                  self.accum.memory_bytes()) / 2 ** 30)
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

    def sky(self):
        with self.grid_lock:
            if self.grid is None:
                return dict(nside=None, hpix=[])
            act = np.nonzero(self.grid.hpix_of >= 0)[0]
            sig = np.sqrt(np.median(self.searcher.var[:, act], axis=0))
            return dict(nside=self.grid.nside, hpix=self.grid.hpix_of[act].tolist(),
                        age_s=(self.searcher.age[act] * self.dt).tolist(),
                        sigma=sig.tolist())
