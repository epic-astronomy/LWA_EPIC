"""
Real-time dedispersion search of every tracked sky pixel.

Per image (83.6 ms at LWA-SV) the regridded sky values [chan, pixel] are

1. normalised per pixel and channel by a running mean and variance
   (time constants norm_tau_s and var_tau_s; the variance from successive
   differences, so slow trends do not inflate it; zero output during warm-up),
2. freed of the common mode: the median over the sky of each channel is
   subtracted, which removes RFI and gain changes seen by the whole image
   while a burst, confined to a beam, leaves the median unchanged,
3. flagged for interference: a channel whose spread over the sky (1.4826
   times the median absolute deviation, about 1 for noise) exceeds
   chan_flag_sigma times that of the band (the median over the channels, at
   least 1) carries interference seen over much of the sky in this
   image, such as a carrier scattered by an aircraft or a meteor trail, and
   is set to zero for every pixel. A channel flagged in more than half of the
   images (exponential rate, time constant flag_tau_s) is masked, set to zero
   in every image, until its rate falls below a fifth. Single cells (pixel
   and channel) with |x| > clip are set to zero as well, unless their sky
   neighbours deviate with them (a source seen through the beam, such as a
   bright burst). Neither a flagged channel nor a cell beyond the clip
   updates the running statistics,
4. pushed into tier 0. Each tier high-passes its input with a time constant
   of hp_tau of its own samples and passes the sum of every two inputs to the
   next tier.

When a tier has collected a block of new samples it removes the mean (or a
linear fit) from each channel of each pixel over its history (max_delay +
block samples, or longer with detrend_len / detrend_factor), and above tier 0
the mean over channels at each time (a zero-DM filter); it runs the FDMT and
the boxcar search (kernels.search) over the last max_delay + block samples,
and returns the best S/N per pixel and segment. Boxcar widths, the high-pass
and the history are set per tier, so that the low tiers search the pulses of
seconds to a minute that scattering makes of FRBs below 100 MHz. The noise of every delay row and
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


def parse_tier_spec(spec: str):
    """
    "0-4:1,2,4,8,16,32,64;5-:1,2,4,8" -> [(0, 4, (1, 2, ...)), (5, None, (1, 2, 4, 8))].
    A spec without a colon applies to every tier ("1,2,4,8" or "32").
    """
    out = []
    for part in str(spec).replace(" ", "").split(";"):
        if not part:
            continue
        if ":" in part:
            rng, vals = part.split(":", 1)
            lo, _, hi = rng.partition("-")
            lo = int(lo)
            hi = (None if hi == "" else int(hi)) if "-" in rng else lo
        else:
            lo, hi, vals = 0, None, part
        out.append((lo, hi, tuple(float(v) for v in vals.split(",") if v)))
    if not out:
        raise ValueError(f"empty tier spec {spec!r}")
    return out


def tier_spec(spec, level: int, per_tier_list: bool = False):
    """
    The value of a per-tier setting for one tier. spec is a number, a spec
    string (parse_tier_spec; one value or a list per tier range), or a
    sequence: of sequences (one per tier, the last repeated above), or of
    numbers, which is the value of every tier (per_tier_list False, as for
    boxcar widths) or one number per tier (per_tier_list True, as for hp_tau).
    A spec string's value is a tuple when it lists several numbers.
    """
    if isinstance(spec, str):
        for lo, hi, vals in parse_tier_spec(spec):
            if level >= lo and (hi is None or level <= hi):
                return vals if len(vals) > 1 else vals[0]
        raise ValueError(f"tier {level} is not covered by {spec!r}")
    if isinstance(spec, (int, float, np.integer, np.floating)):
        return spec
    seq = list(spec)
    if seq and isinstance(seq[0], (list, tuple, np.ndarray)):
        return tuple(seq[min(level, len(seq) - 1)])
    if per_tier_list:
        return seq[min(level, len(seq) - 1)]
    return tuple(seq)


def boxcar_stride(w: int, div: int, block: int) -> int:
    """The evaluation stride of a boxcar of w samples (Searcher stride_div)"""
    s = 1
    if div > 0:
        while 2 * s <= w // div and block % (2 * s) == 0:
            s *= 2
    return s


# effective number of independent boxcar trials per delay row and sample,
# relative to width 1, falls as width ** -TRIAL_EXP (white noise through the
# tier chain, measured with the search kernel; see far_threshold)
TRIAL_EXP = 1.0
REF_WIDTHS = (1, 2, 4, 8)


def trial_rate(tiers, D: int, widths_of=None, exp: float = TRIAL_EXP) -> float:
    """Effective trials per second and pixel of a set of tiers"""
    r = 0.0
    for t in tiers:
        ws = t.widths if widths_of is None else widths_of(t)
        r += (D - t.row0 + 1) / t.dt * sum(float(w) ** -exp for w in ws)
    return r


def far_threshold(tiers, threshold_ref: float, D: int, ref_widths=REF_WIDTHS) -> float:
    """
    The S/N threshold whose Gaussian false-alarm rate with the tiers' boxcars
    equals that of threshold_ref with ref_widths in every tier:
    R Q(s) = R_ref Q(threshold_ref), R being trial_rate and Q the upper tail
    of the unit normal.
    """
    r_new = trial_rate(tiers, D)
    r_ref = trial_rate(tiers, D, lambda t: ref_widths)
    q = lambda x: 0.5 * math.erfc(x / math.sqrt(2.0))
    target = r_ref * q(threshold_ref) / r_new
    lo, hi = threshold_ref - 3.0, threshold_ref + 3.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if q(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


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
                 hp_tau=32.0, clip=12.0, median_npix=2048, unix0=0.0, rng=None,
                 chan_flag_sigma=1.6, flag_tau_s=30.0, mask_on=0.5, mask_off=0.2,
                 hold_s=30.0, clip_nbr_min=3, clip_nbr_frac=0.3, hp_factor=4.0,
                 detrend_len=0, detrend_factor=0.0, detrend="linear", far_matched=False,
                 stride_div=0, noise_every=1, settle_hp=3.0, narrow_max=8,
                 history_dtype="float32"):
        """
        widths: boxcar widths in tier samples, for every tier (a sequence of
        ints), per tier (a sequence of sequences, the last repeated for the
        tiers above), or a spec string such as "0-4:1,2,4,8,16,32,64;5-:1,2,4,8"
        (tier_spec).
        hp_tau: time constant of each tier's high-pass, in its own samples: the
        smallest for every tier (float), or per tier (sequence or spec). A tier
        uses at least hp_factor times its widest boxcar, so that a pulse of
        that width loses at most about 1 / (2 hp_factor) of its signal.
        detrend_len, detrend_factor: each tier keeps at least detrend_len
        samples (int, sequence or spec) and at least detrend_factor times its
        widest boxcar, besides the max_delay + block samples of the transform;
        the trend is fitted over the whole history, the transform runs over the
        last max_delay + block samples. detrend: "linear" (a straight line per
        channel and pixel) or "mean".
        stride_div: when > 0, a boxcar of w samples is evaluated at every
        s-th end sample only, s the largest power of two up to w / stride_div
        that divides the tier's block: a pulse matching the boxcar then loses
        at most s / (2 w) of its signal, 1 / (4 stride_div) on average, for
        about 1 / s of the boxcar's cost.
        settle_hp, narrow_max: a boxcar wider than narrow_max samples reports
        a pixel only once the pixel has been tracked for the tier's history
        plus settle_hp high-pass time constants (in images: warm-up + (W +
        settle_hp hp_tau) decim). A pixel that has just risen above the
        elevation limit brightens quickly while its running mean and the
        tier's high-pass start from nothing, which a wide boxcar at low DM
        takes for a broadband pulse; the narrow boxcars are not affected.
        noise_every: the noise of every row and width is measured in one
        block of this many (always until a tier is calibrated); the noise
        sums take about a third of the boxcar search.
        threshold: S/N threshold. With far_matched it is the threshold of the
        reference search (boxcars 1, 2, 4, 8 in every tier) and is raised so
        that the Gaussian false-alarm rate stays that of the reference
        (far_threshold); self.threshold is the threshold applied.
        clip: a normalised cell (one pixel, channel and image) beyond this is
        set to zero and kept out of the statistics, unless it is sky-shaped:
        at least clip_nbr_min of its HEALPix neighbours (set_neighbours)
        deviate the same way by clip_nbr_frac of its deviation in the same
        channel and image. A burst of S/N s whose channel signal falls into
        one image adds s sqrt(n_eff / C) to that cell, n_eff (about 1.7 at the
        top of a tier) being the samples per channel of the S/N definition:
        12 is reached at S/N 73 for C = 64 and 104 for C = 128. Such a burst
        lights its neighbours through the beam and is kept (withheld from the
        statistics only); a glitch of a single cell is still set to zero.
        chan_flag_sigma: spread over the sky (robust standard deviation of the
        normalised values, about 1 for noise), relative to the median spread
        of the channels (at least 1), above which a channel is flagged in one
        image. A rise of noise over the whole band is not flagged; it is
        searched and recorded in the conditions.
        flag_tau_s, mask_on, mask_off: time constant of the flag rate of each
        channel, and the rates at which a channel is masked and released.
        hold_s: longest time for which the statistics of a flagged or clipped
        cell are withheld; after it they follow again (skymon_normalise).
        history_dtype: "float32", or "float16" to keep the tiers' histories in
        half precision: half the memory and memory traffic, with every sum
        formed in single precision (a stored sample carries a relative
        rounding error of at most 2^-11 against unit noise).
        """
        self.edges = np.asarray(edges_mhz, float)
        self.C = len(self.edges) - 1
        self.P = capacity
        self.dt = dt_img
        self.D = max_delay or self.C
        self.history_dtype = np.dtype(history_dtype)
        if self.history_dtype not in (np.float32, np.float16):
            raise ValueError(f"history_dtype must be float32 or float16, not {history_dtype}")
        self.seg = seg
        self.threshold_ref = threshold
        self.threshold = threshold
        self.detrend_mode = {"linear": 1, "mean": 2}[detrend]
        self.noise_every = max(1, int(noise_every))
        self.nthreads = nthreads
        self.warmup = warmup_images
        self.a_mu = dt_img / norm_tau_s
        self.a_var = dt_img / var_tau_s
        self.clip = clip
        self.clip_nbr_min, self.clip_nbr_frac = clip_nbr_min, clip_nbr_frac
        self.nbr = None  # [P, 8] slots of the sky neighbours (set_neighbours)
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
        # interference flagging (skymon_normalise)
        self.chan_flag_sigma = chan_flag_sigma
        self.a_flag = dt_img / flag_tau_s
        self.mask_on, self.mask_off = mask_on, mask_off
        self.max_hold = int(min(65535, round(hold_s / dt_img)))
        self.held = np.zeros((C, P), np.uint16)
        self.flags = np.zeros(C, np.uint8)  # flagged in the last image
        self.spread = np.zeros(C, np.float32)  # spread over the sky in the last image
        self.flag_rate = np.zeros(C, np.float64)
        self.chan_mask = np.zeros(C, np.uint8)  # masked: set to zero in every image
        self.flag_total = np.zeros(C, np.int64)  # images flagged, per channel
        self.n_minute = max(1, int(round(60.0 / dt_img)))
        for t in self.tiers:
            ws = np.atleast_1d(np.asarray(tier_spec(widths, t.level)))
            # a boxcar ends in the block and starts within the window: widths
            # beyond max_delay + 1 samples (small test bands) are left out
            ws = sorted(set(int(w) for w in ws if 1 <= int(w) <= self.D + 1))
            if not ws:
                raise ValueError(f"tier {t.level}: no boxcar width in 1..{self.D + 1}")
            t.widths = np.asarray(ws, np.int32)
            wmax = int(t.widths.max())
            hp = max(float(tier_spec(hp_tau, t.level, True)), hp_factor * wmax)
            t.hp_tau = hp
            t.a_hp = 1.0 / hp
            span = max(int(tier_spec(detrend_len, t.level, True)),
                       int(math.ceil(detrend_factor * wmax)))
            W = max(self.D + t.block, span)
            t.f0 = W - (self.D + t.block)  # samples before the transform's window
            t.history = np.zeros((C, W, P), self.history_dtype)
            t.base = np.zeros((C, P), np.float32)
            t.acc = np.zeros((C, P), np.float32)
            t.acc_n = 0
            # next write position in the window; the first block of each tier
            # is shortened by a different amount so the tiers' searches fall
            # on different images instead of all on the same one
            t.pos = W - t.block + (t.level * t.block // len(self.tiers)) % t.block
            t.pushed = 0  # samples pushed in total
            t.runs = 0
            t.calibrated = False
            t.sigma = np.ones((self.D + 1, len(t.widths)), np.float32)
            t.strides = np.array([boxcar_stride(int(w), stride_div, t.block) for w in t.widths],
                                 np.int32)
            t.settle = int(warmup_images + math.ceil((W + settle_hp * hp) * t.decim))
            t.narrow_max = int(narrow_max)
        # every width searched in some tier, and tier 0's high-pass weight
        self.widths = np.asarray(sorted(set(int(w) for t in self.tiers for w in t.widths)),
                                 np.int32)
        self.a_hp = self.tiers[0].a_hp
        self.a_hp_tiers = np.array([t.a_hp for t in self.tiers], np.float32)
        if far_matched:
            self.threshold = far_threshold(self.tiers, threshold, self.D)
        self.images = 0
        # channels set to zero (flagged or masked) in each recent image, for
        # the flagged fraction of the last minute and of a candidate's sweep
        span = max(t.history.shape[1] * t.decim for t in self.tiers)
        self.zeroed_hist = np.zeros(max(self.n_minute, span) + 1, np.int16)
        self.stats = dict(search_s=0.0, ingest_s=0.0, blocks=0)

    # ------------------------------------------------------------------ slots
    def set_live(self, live: np.ndarray):
        self.live[:] = live.astype(np.uint8)
        idx = np.nonzero(self.live)[0]
        n = min(self.median_npix, len(idx))
        self.sub = np.sort(self.rng.choice(idx, n, replace=False)).astype(np.int32) if n else \
            np.zeros(0, np.int32)

    def set_neighbours(self, nbr: np.ndarray | None):
        """The slots of each slot's sky neighbours [P, 8] (-1: none), for the clip"""
        self.nbr = None if nbr is None else np.ascontiguousarray(nbr, np.int32)

    def reset_slots(self, slots):
        slots = np.asarray(slots, np.int64)
        if len(slots) == 0:
            return
        self.mu[:, slots] = 0
        self.var[:, slots] = 0
        self.prev[:, slots] = 0
        self.age[slots] = 0
        self.held[:, slots] = 0
        for t in self.tiers:
            t.history[:, :, slots] = 0
            t.base[:, slots] = 0
            t.acc[:, slots] = 0

    def rebase(self, slots, mean, first):
        """
        New baselines after a step of the data (a change of the calibration):
        the running mean of each channel of the given slots becomes mean
        [C, len(slots)], the mean of the images after the step, and the
        previous raw value (for the successive differences) first [C,
        len(slots)], the first of them. Slots still warming up are left alone.
        """
        slots = np.asarray(slots, np.int64)
        warm = self.age[slots] > self.warmup
        s_ = slots[warm]
        self.mu[:, s_] = mean[:, warm]
        self.prev[:, s_] = first[:, warm]

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
        k = self.images % len(self.zeroed_hist)
        if raw is None:
            if getattr(self, "_zeros", None) is None:
                self._zeros = np.zeros((self.C, self.P), np.float32)
            x = self._zeros
            self.zeroed_hist[k] = 0
        else:
            kernels.normalise(raw, self.mu, self.var, self.prev, self.age, self.live, acc, self.sub,
                              self.warmup, self.a_mu, self.a_var, self.clip, self.median,
                              self.nthreads, chan_flag_sigma=self.chan_flag_sigma,
                              chan_mask=self.chan_mask, flags=self.flags, spread=self.spread,
                              held=self.held, max_hold=self.max_hold, nbr=self.nbr,
                              nbr_min=self.clip_nbr_min, nbr_frac=self.clip_nbr_frac)
            self.zeroed_hist[k] = np.count_nonzero(self.flags | self.chan_mask)
            self._update_mask()
            x = raw
        self.images += 1
        self.stats["ingest_s"] += time.perf_counter() - t0
        return self._push_tier(0, x)

    def _update_mask(self):
        """
        Persistent interference: the flag rate of each channel follows the
        flags with time constant flag_tau_s. A channel is masked when its rate
        exceeds mask_on and released when it falls below mask_off; between
        the two it keeps its state, so a channel whose interference comes and
        goes is not switched on and off with every gap. The new mask applies
        from the next image.
        """
        self.flag_total += self.flags
        self.flag_rate += self.a_flag * (self.flags - self.flag_rate)
        self.chan_mask[self.flag_rate > self.mask_on] = 1
        self.chan_mask[self.flag_rate < self.mask_off] = 0

    def _push_tier(self, j: int, x: np.ndarray):
        """
        Push x into tier j and, through the decimation sums, into the tiers
        above it that complete a sample; search every tier whose window is
        full. Tier j passes the sum of every two of its inputs, times
        1/sqrt(2), to tier j + 1. One compiled call does all the pushes
        (kernels.tier_cascade); the searches then run from the highest tier
        down, and the results are returned from tier j up.
        """
        n = len(self.tiers)
        k = j + 1  # tiers j .. k - 1 receive a sample
        while k < n and self.tiers[k - 1].acc_n == 1:
            k += 1
        reached = self.tiers[j:k]
        kernels.tier_cascade(x, reached, self.a_hp_tiers[j:k],
                             float(np.float32(1 / math.sqrt(2))),
                             last_dec=k < n, nthreads=self.nthreads)
        for i, tier in enumerate(reached):
            tier.pos += 1
            tier.pushed += 1
            if j + i + 1 < n:
                tier.acc_n = 0 if i + 1 < len(reached) else tier.acc_n + 1
        out = []
        for tier in reversed(reached):
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
        measure = not tier.calibrated or tier.runs % self.noise_every == 0
        best, info, sumsq, cnt = kernels.search(
            tier.history, self.plan, self.D, tier.row0, tier.block, tier.widths, tier.sigma,
            self.seg, 5.0 if tier.calibrated else 1e30, self.nthreads,
            detrend=self.detrend_mode, zerodm=tier.level > 0, f0=tier.f0, strides=tier.strides,
            noise=measure)
        with np.errstate(invalid="ignore", divide="ignore"):
            s_new = np.sqrt(sumsq / np.maximum(cnt, 1)).astype(np.float32)
        s_new[cnt == 0] = 0
        ok = s_new > 0
        if not measure:
            pass
        elif not tier.calibrated:
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
                w = int(tier.widths[wi])
                if w > tier.narrow_max and self.age[p] < tier.settle:
                    continue  # a wide boxcar on a pixel that has just risen
                t_end = first + tier.f0 + int(t)
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
            W, B = tier.history.shape[1], tier.block
            kernels.shift(tier.history, 0, B, W - B, self.nthreads)
            tier.pos = W - B
        self._pending_shift = []

    # ------------------------------------------------------------- utilities
    def zeroed_fraction(self, first_image: int, n_images: int):
        """
        Fraction of the channels set to zero (flagged or masked) over images
        first_image .. first_image + n_images - 1 (absolute image indices);
        None when none of them is still in the history.
        """
        lo = max(first_image, self.images - len(self.zeroed_hist) + 1, 0)
        hi = min(first_image + n_images, self.images)
        if hi <= lo:
            return None
        idx = np.arange(lo, hi) % len(self.zeroed_hist)
        return float(self.zeroed_hist[idx].sum()) / (self.C * (hi - lo))

    def tier_zeroed_fraction(self, tier: Tier, first_sample: int, last_sample: int):
        """zeroed_fraction over tier samples first_sample..last_sample (inclusive)"""
        return self.zeroed_fraction(first_sample * tier.decim,
                                    (last_sample - first_sample + 1) * tier.decim)

    def rfi_status(self) -> dict:
        """Interference flagging: the last minute, and the channels masked now"""
        n = min(self.images, self.n_minute)
        frac = self.zeroed_fraction(self.images - n, n) if n else None
        return dict(flagged_fraction_last_minute=None if frac is None else round(frac, 5),
                    masked_channels=np.nonzero(self.chan_mask)[0].tolist(),
                    flagged_last_image=np.nonzero(self.flags)[0].tolist(),
                    flagged_images_per_channel=self.flag_total.tolist(),
                    chan_flag_sigma=self.chan_flag_sigma, clip=self.clip)

    def describe(self):
        return [dict(level=t.level, dt_s=t.dt, dm_min=t.dm_min, dm_max=t.dm_max,
                     dm_step=t.dm_per_delay, block=t.block, window=self.D + t.block,
                     history=int(t.history.shape[1]), widths=t.widths.tolist(),
                     max_width_s=float(t.widths.max() * t.dt), hp_tau=t.hp_tau,
                     strides=t.strides.tolist(),
                     latency_s=t.block * t.dt) for t in self.tiers]

    def white_sigma(self, tier: Tier, npix: int = 32, seed: int = 0) -> np.ndarray:
        """
        The noise [D + 1, nw] of every delay row and boxcar of a tier for unit
        white noise in every image, through the same chain as the data (the
        running mean, the tier's high-pass, the trend fit, the zero-DM filter,
        the transform and boxcars): the theory against which the measured
        noise (tier.sigma) is compared. Computed once per tier, on npix
        pixels and about 2000 tier samples.
        """
        cache = getattr(self, "_white", None)
        if cache is None:
            cache = self._white = {}
        if tier.level in cache:
            return cache[tier.level]
        rng = np.random.default_rng(seed + tier.level)
        C, B = self.C, tier.block
        W = tier.history.shape[1]
        nwin = max(2, int(math.ceil(2048 / B)))
        skip = int(4 * max(tier.hp_tau, 1.0 / (self.a_mu * tier.decim) / 8)) + 64
        N = skip + W + (nwin - 1) * B
        x = rng.standard_normal((N, C, npix)).astype(np.float32)
        a_mu = np.float32(min(1.0, self.a_mu * tier.decim))
        a_hp = np.float32(tier.a_hp)
        m = np.zeros((C, npix), np.float32)
        b = np.zeros((C, npix), np.float32)
        for t in range(N):  # running mean out (normalisation), then the tier's high-pass
            v = x[t] - m
            m += a_mu * v
            y = v - b
            b += a_hp * y
            x[t] = y
        x = np.ascontiguousarray(x.transpose(1, 0, 2))
        ss = np.zeros((self.D + 1, len(tier.widths)))
        cn = np.zeros_like(ss)
        ones = np.ones_like(tier.sigma)
        for k in range(nwin):
            s0 = skip + k * B
            _, _, sumsq, cnt = kernels.search(
                np.ascontiguousarray(x[:, s0:s0 + W]), self.plan, self.D, tier.row0, B,
                tier.widths, ones, B, 1e30, self.nthreads, detrend=self.detrend_mode,
                zerodm=tier.level > 0, f0=tier.f0, strides=tier.strides)
            ss += sumsq
            cn += cnt
        with np.errstate(invalid="ignore", divide="ignore"):
            sig = np.where(cn > 0, np.sqrt(ss / np.maximum(cn, 1)), np.nan).astype(np.float32)
        cache[tier.level] = sig
        return sig

    def noise_probes(self, widths_s=(1.0, 7.0, 20.0)):
        """
        For each target width (s), the tier and boxcar of nearest duration (in
        log; the lowest tier among equals): [(target_s, tier, width index,
        width_s)].
        """
        out = []
        for ws in widths_s:
            best = None
            for t in self.tiers:
                for wi, w in enumerate(t.widths):
                    d = abs(math.log(w * t.dt / ws))
                    if best is None or d < best[0] - 1e-9:
                        best = (d, t, wi, float(w * t.dt))
            out.append((float(ws), best[1], best[2], best[3]))
        return out

    def noise_at_widths(self, widths_s=(1.0, 7.0, 20.0)) -> list:
        """
        The measured search noise relative to white noise at the given widths:
        for each, the median over the tier's reported delay rows of
        tier.sigma / white_sigma, at the nearest tier and boxcar (noise_probes).
        None until the tier has measured its noise on a full window.
        """
        out = []
        for ws, t, wi, w_s in self.noise_probes(widths_s):
            ratio = None
            if t.calibrated:
                ref = self.white_sigma(t)[t.row0:, wi]
                got = t.sigma[t.row0:, wi]
                ok = np.isfinite(ref) & (ref > 0) & (got > 0)
                if ok.any():
                    ratio = round(float(np.median(got[ok] / ref[ok])), 4)
            out.append(dict(target_s=ws, tier=t.level, width=int(t.widths[wi]),
                            width_s=round(w_s, 3), ratio=ratio))
        return out

    def tier_for_dm(self, dm: float) -> Tier:
        for t in self.tiers:
            if dm <= t.dm_max:
                return t
        return self.tiers[-1]

    def memory_bytes(self) -> int:
        n = self.mu.nbytes + self.var.nbytes + self.prev.nbytes + self.held.nbytes
        for t in self.tiers:
            n += t.history.nbytes + t.base.nbytes + t.acc.nbytes
        return n
