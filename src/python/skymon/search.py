"""
Real-time dedispersion search of every tracked sky pixel.

Per image (83.6 ms at LWA-SV) the regridded sky values [chan, pixel] are

1. normalised per pixel and channel by a running mean and variance
   (time constants norm_tau_s and var_tau_s; the variance from successive
   differences, so slow trends do not inflate it; zero output during warm-up),
2. freed of the common mode: the median over the sky of each channel is
   subtracted, which removes RFI and gain changes seen by the whole image
   while a burst, confined to a beam, leaves the median unchanged,
3. pushed into tier 0. Each tier high-passes its input with a time constant
   of hp_tau of its own samples and passes the sum of every two inputs to the
   next tier.

When a tier has collected a block of new samples it removes a linear fit from
each channel of each pixel over its window of max_delay + block samples (and,
above tier 0, the mean over channels at each time: a zero-DM filter), runs
the FDMT and boxcar search (kernels.search) over the window, and
returns the best S/N per pixel and segment. The noise of every delay row and
boxcar width is measured from the data itself (clipped RMS of the previous
blocks), so the S/N is calibrated without assuming white noise.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import kernels
from .dedisp import K_DM, Tier, fdmt_plan, make_tiers


@dataclass
class Hit:
    tier: int
    slot: int
    row: int  # delay row
    t_end: int  # absolute tier sample of the boxcar's last sample
    width: int  # boxcar width, tier samples
    snr: float
    dm: float
    t_bottom: float  # unix arrival at the bottom band edge
    t_inf: float  # unix arrival at infinite frequency
    width_s: float


@dataclass
class BlockResult:
    tier: Tier
    hits: list
    window: np.ndarray | None  # [C, W, P] view of the window searched (valid until next push)
    first_sample: int  # absolute tier sample of window[:, 0]
    elapsed: float
    calibrating: bool
    sigma: np.ndarray = field(repr=False, default=None)


class Searcher:
    def __init__(self, edges_mhz, dt_img, capacity, dm_max=3000.0, widths=(1, 2, 4, 8),
                 max_delay=None, block0=256, min_block=32, seg=16, threshold=8.0,
                 nthreads=8, warmup_images=64, norm_tau_s=120.0, var_tau_s=300.0,
                 hp_tau=32.0, clip=20.0, median_npix=2048, unix0=0.0, rng=None):
        self.edges = np.asarray(edges_mhz, float)
        self.C = len(self.edges) - 1
        self.P = capacity
        self.dt = dt_img
        self.D = max_delay or self.C
        self.widths = np.asarray(widths, np.int32)
        self.seg = seg
        self.threshold = threshold
        self.nthreads = nthreads
        self.warmup = warmup_images
        self.a_mu = dt_img / norm_tau_s
        self.a_var = dt_img / var_tau_s
        self.a_hp = 1.0 / hp_tau
        self.clip = clip
        self.unix0 = unix0
        self.rng = rng or np.random.default_rng(0)
        self.median_npix = median_npix
        self.f_lo, self.f_hi = float(self.edges[0]), float(self.edges[-1])
        self.tiers = make_tiers(self.f_lo, self.f_hi, dt_img, self.C, dm_max, self.D, block0,
                                min_block)
        self.plan = kernels.flatten_plan(fdmt_plan(self.edges, self.D))
        C, P = self.C, self.P
        self.mu = np.zeros((C, P), np.float32)
        self.var = np.zeros((C, P), np.float32)
        self.prev = np.zeros((C, P), np.float32)
        self.age = np.zeros(P, np.int32)
        self.live = np.zeros(P, np.uint8)
        self.sub = np.zeros(0, np.int32)
        self.median = np.zeros(C, np.float32)
        for t in self.tiers:
            W = self.D + t.block
            t.history = np.zeros((C, W, P), np.float32)
            t.base = np.zeros((C, P), np.float32)
            t.acc = np.zeros((C, P), np.float32)
            t.acc_n = 0
            # next write position in the window; the first block of each tier
            # is shortened by a different amount so the tiers' searches fall
            # on different images instead of all on the same one
            t.pos = self.D + (t.level * t.block // len(self.tiers)) % t.block
            t.pushed = 0  # samples pushed in total
            t.runs = 0
            t.calibrated = False
            t.sigma = np.ones((self.D + 1, len(self.widths)), np.float32)
        self.images = 0
        self.stats = dict(search_s=0.0, ingest_s=0.0, blocks=0)

    # ------------------------------------------------------------------ slots
    def set_live(self, live: np.ndarray):
        self.live[:] = live.astype(np.uint8)
        idx = np.nonzero(self.live)[0]
        n = min(self.median_npix, len(idx))
        self.sub = np.sort(self.rng.choice(idx, n, replace=False)).astype(np.int32) if n else \
            np.zeros(0, np.int32)

    def reset_slots(self, slots):
        slots = np.asarray(slots, np.int64)
        if len(slots) == 0:
            return
        self.mu[:, slots] = 0
        self.var[:, slots] = 0
        self.prev[:, slots] = 0
        self.age[slots] = 0
        for t in self.tiers:
            t.history[:, :, slots] = 0
            t.base[:, slots] = 0
            t.acc[:, slots] = 0

    # ------------------------------------------------------------- per image
    def tier_time(self, tier: Tier, sample: float) -> float:
        """Unix start time of tier sample index (fractional allowed)"""
        return self.unix0 + sample * tier.dt

    def push(self, raw: np.ndarray | None, acc: np.ndarray | None = None):
        """
        Process the next image. raw [C, P] float32 (modified in place), or None
        for a missing image (zeros are searched, statistics are not updated).
        Returns a list of BlockResult for the tiers that completed a block.
        """
        t0 = time.perf_counter()
        if raw is None:
            x = np.zeros((self.C, self.P), np.float32)
        else:
            kernels.normalise(raw, self.mu, self.var, self.prev, self.age, self.live, acc, self.sub,
                              self.warmup, self.a_mu, self.a_var, self.clip, self.median,
                              self.nthreads)
            x = raw
        self.images += 1
        self.stats["ingest_s"] += time.perf_counter() - t0
        return self._push_tier(0, x)

    def _push_tier(self, j: int, x: np.ndarray):
        tier = self.tiers[j]
        nxt = j + 1 < len(self.tiers)
        kernels.tier_push(x, tier.base, tier.history, tier.pos, self.a_hp,
                          tier.acc if nxt else None, self.nthreads)
        tier.pos += 1
        tier.pushed += 1
        out = []
        if nxt:
            tier.acc_n += 1
            if tier.acc_n == 2:
                y = tier.acc * np.float32(1 / math.sqrt(2))
                tier.acc[:] = 0
                tier.acc_n = 0
                out += self._push_tier(j + 1, y)
        if tier.pos == tier.history.shape[1]:
            out.insert(0, self._search(tier))
        return out

    def _search(self, tier: Tier) -> BlockResult:
        t0 = time.perf_counter()
        W = tier.history.shape[1]
        first = tier.pushed - W
        # the noise is measured once the window holds only data, never start-up zeros
        full = tier.pushed >= W + tier.block
        # hits are reported only with a noise estimate from an earlier full window
        calibrating = not (full and tier.calibrated)
        best, info, sumsq, cnt = kernels.search(
            tier.history, self.plan, self.D, tier.row0, tier.block, self.widths, tier.sigma,
            self.seg, 5.0 if tier.calibrated else 1e30, self.nthreads,
            zerodm=tier.level > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            s_new = np.sqrt(sumsq / np.maximum(cnt, 1)).astype(np.float32)
        s_new[cnt == 0] = 0
        ok = s_new > 0
        if not tier.calibrated:
            tier.sigma = np.where(ok, s_new, 1.0).astype(np.float32)
            tier.calibrated = full
        else:
            mix = np.sqrt(0.7 * tier.sigma ** 2 + 0.3 * s_new ** 2)
            tier.sigma = np.where(ok, mix, tier.sigma).astype(np.float32)
        hits = []
        if not calibrating:
            ps, segs = np.nonzero(best > self.threshold)
            for p, sg in zip(ps, segs):
                row, t, wi = info[p, sg]
                w = int(self.widths[wi])
                t_end = first + int(t)
                dm = row * tier.dm_per_delay
                # bottom-edge arrival in the middle of the boxcar
                t_bottom = self.tier_time(tier, t_end - (w - 1) / 2.0 + 0.5)
                hits.append(Hit(tier.level, int(p), int(row), t_end, w, float(best[p, sg]),
                                float(dm), t_bottom, t_bottom - K_DM * dm / self.f_lo ** 2,
                                w * tier.dt))
        tier.runs += 1
        el = time.perf_counter() - t0
        self.stats["search_s"] += el
        self.stats["blocks"] += 1
        res = BlockResult(tier, hits, tier.history, first, el, calibrating, tier.sigma)
        self._pending_shift = getattr(self, "_pending_shift", [])
        self._pending_shift.append(tier)
        return res

    def finish_block(self):
        """Shift the searched windows; call after the BlockResults have been used"""
        for tier in getattr(self, "_pending_shift", []):
            D, B = self.D, tier.block
            tier.history[:, :D] = tier.history[:, B:B + D]
            tier.pos = D
        self._pending_shift = []

    # ------------------------------------------------------------- utilities
    def describe(self):
        return [dict(level=t.level, dt_s=t.dt, dm_min=t.dm_min, dm_max=t.dm_max,
                     dm_step=t.dm_per_delay, block=t.block, window=self.D + t.block,
                     latency_s=t.block * t.dt) for t in self.tiers]

    def tier_for_dm(self, dm: float) -> Tier:
        for t in self.tiers:
            if dm <= t.dm_max:
                return t
        return self.tiers[-1]

    def memory_bytes(self) -> int:
        n = self.mu.nbytes + self.var.nbytes + self.prev.nbytes
        for t in self.tiers:
            n += t.history.nbytes + t.base.nbytes + t.acc.nbytes
        return n
