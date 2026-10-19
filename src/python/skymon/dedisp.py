"""
Streaming incoherent dedispersion of many sky pixels at once.

The search runs in tiers. Tier j sees the data averaged over 2**j images and
searches the delays 0..D samples (D = max_delay, nchan by default) at that
resolution with the fast dispersion measure transform (FDMT, Zackay & Ofek
2017, ApJ 835, 11). Because the dispersion delay is linear in DM, every tier
has the same delay table: tier j simply covers DMs 2**j times larger than
tier 0. Tier j > 0 only reports delays above D/2, which tier j - 1 cannot
reach, so the tiers together cover the DM range in octaves.

The smearing within a channel is the band sweep divided by the number of
channels, so at the top of every tier a dispersed pulse spans about D / nchan
samples per channel. Averaging further would lose signal; averaging less
would waste compute. This keeps the history needed per tier at about
D + block samples, independent of DM (the reason the search fits in RAM).

Arrays keep the pixel axis last and contiguous: every numpy operation acts on
(time x pixel) slabs, so the Python overhead is amortised over many pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

K_DM = 4.148808e3  # s MHz^2 / (pc cm^-3)


def sweep_per_dm(f_lo_mhz: float, f_hi_mhz: float) -> float:
    """Delay (s) between the band edges per unit DM"""
    return K_DM * (f_lo_mhz ** -2 - f_hi_mhz ** -2)


def fdmt_plan(f_edges: np.ndarray, max_delay: int):
    """
    Precompute the FDMT merge plan for channel edges (MHz, ascending, any
    number of channels) and a total delay of max_delay samples across the band.

    Returns (init_nd, levels): init_nd[c] is the number of extra delays inside
    channel c; levels is a list of merge levels, each a list of
    (nd, d_lo[], d_up[]) for every subband of the level, in band order.
    Adjacent subbands are merged in pairs. When a level has an odd number of
    subbands, its lowest subband is passed on unmerged, as (nd, None, None):
    192 channels form three subbands of 64, of which the upper two are merged
    and the lowest joins them in the last merge. The lowest subband has the
    most delays, so passing it on rather than another saves the most work.
    The delays of a merge are split in proportion to the dispersion delays of
    its halves, whatever their widths, so the transform is exact to the same
    rounding as with a power of two.
    """
    f_edges = np.asarray(f_edges, dtype=np.float64)
    n_chan = len(f_edges) - 1
    if n_chan < 1:
        raise ValueError("FDMT needs at least one channel")
    inv2 = f_edges ** -2
    span = inv2[0] - inv2[-1]

    def n_delay(i0, i1):
        return int(np.ceil(max_delay * (inv2[i0] - inv2[i1]) / span - 1e-9))

    init_nd = [n_delay(c, c + 1) for c in range(n_chan)]
    bands = [(c, c + 1, init_nd[c]) for c in range(n_chan)]
    levels = []
    while len(bands) > 1:
        merged, level = [], []
        odd = len(bands) % 2
        if odd:
            level.append((bands[0][2], None, None))
            merged.append(bands[0])
        for (lo0, lo1, nd_lo), (up0, up1, nd_up) in zip(bands[odd::2], bands[odd + 1::2]):
            nd = n_delay(lo0, up1)
            frac = (inv2[lo0] - inv2[lo1]) / (inv2[lo0] - inv2[up1])
            d = np.arange(nd + 1)
            d_lo = np.minimum(np.rint(d * frac).astype(int), nd_lo)
            d_up = np.minimum(d - d_lo, nd_up)
            level.append((nd, d_lo, d_up))
            merged.append((lo0, up1, nd))
        levels.append(level)
        bands = merged
    return init_nd, levels


def fdmt(data: np.ndarray, plan) -> np.ndarray:
    """
    data: [n_chan, n_time, n_pix] float32, channels ascending in frequency.
    Returns [max_delay + 1, n_time, n_pix]. Row d sums the dispersion track whose
    arrival at the bottom band edge is d samples after the top edge, indexed by
    the arrival time at the bottom edge. The first d samples of row d are partial
    sums (the caller discards them).
    """
    init_nd, levels = plan
    n_chan, n_time, n_pix = data.shape
    subs = []
    for c in range(n_chan):
        nd = init_nd[c]
        a = np.empty((nd + 1, n_time, n_pix), dtype=np.float32)
        a[0] = data[c]
        for d in range(1, nd + 1):  # within a channel the track spans d + 1 samples
            a[d] = a[d - 1]
            a[d, d:] += data[c, :-d]
        subs.append(a)
    for level in levels:
        merged = []
        odd = len(subs) % 2
        if odd:  # the lowest subband passes on unmerged
            merged.append(subs[0])
        for (nd, d_lo, d_up), a_lo, a_up in zip(level[odd:], subs[odd::2], subs[odd + 1::2]):
            a = np.empty((nd + 1, n_time, n_pix), dtype=np.float32)
            for d in range(nd + 1):
                lo, up = d_lo[d], d_up[d]
                # the lower half is reached last, so its arrival time is the
                # pair's; the upper half is reached lo samples earlier
                if lo:
                    a[d, :lo] = a_lo[lo, :lo]
                    np.add(a_lo[lo, lo:], a_up[up, :-lo], out=a[d, lo:])
                else:
                    np.add(a_lo[0], a_up[up], out=a[d])
            merged.append(a)
        subs = merged
    return subs[0]


def brute_force(data: np.ndarray, f_edges: np.ndarray, max_delay: int) -> np.ndarray:
    """Reference shift-and-add over the same delays (nearest sample, channel centres)"""
    f_edges = np.asarray(f_edges, dtype=np.float64)
    inv2 = f_edges ** -2
    fc2 = (0.5 * (f_edges[:-1] + f_edges[1:])) ** -2
    n_chan, n_time, n_pix = data.shape
    out = np.zeros((max_delay + 1, n_time, n_pix), dtype=np.float64)
    for d in range(max_delay + 1):
        # delay of each channel centre after the top edge, relative to the bottom edge
        sh = np.rint(d * (fc2 - inv2[-1]) / (inv2[0] - inv2[-1])).astype(int)
        lag = d - sh  # samples before the bottom-edge arrival
        for c in range(n_chan):
            if lag[c]:
                out[d, lag[c]:] += data[c, :-lag[c]]
            else:
                out[d] += data[c]
    return out


@dataclass
class Tier:
    """One octave of the DM search, with its own time resolution and history"""
    level: int
    decim: int  # images per sample
    dt: float  # s per sample
    block: int  # new samples per FDMT run
    max_delay: int
    dm_per_delay: float  # DM per sample of delay
    history: np.ndarray = field(repr=False, default=None)  # [chan, max_delay + block, pix]
    n_filled: int = 0  # samples in history
    acc: np.ndarray = field(repr=False, default=None)  # partial decimation sum [chan, pix]
    acc_n: int = 0
    t0_sample: int = 0  # absolute tier-sample index of history[:, 0]

    @property
    def dm_min(self) -> float:
        return 0.0 if self.level == 0 else self.dm_per_delay * (self.max_delay // 2 + 1)

    @property
    def dm_max(self) -> float:
        return self.dm_per_delay * self.max_delay

    @property
    def row0(self) -> int:
        """First delay row this tier reports"""
        return 0 if self.level == 0 else self.max_delay // 2 + 1


def make_tiers(f_lo: float, f_hi: float, dt: float, n_chan: int, dm_max: float,
               max_delay: int | None = None, block0: int = 256, min_block: int = 32):
    """Tiers 0..n covering DM 0..dm_max for a band (MHz edges) sampled every dt seconds"""
    max_delay = max_delay or n_chan
    s = sweep_per_dm(f_lo, f_hi)
    tiers, j = [], 0
    while True:
        tdt = dt * 2 ** j
        tier = Tier(level=j, decim=2 ** j, dt=tdt, block=max(min_block, block0 >> j),
                    max_delay=max_delay, dm_per_delay=tdt / s)
        tiers.append(tier)
        if tier.dm_max >= dm_max:
            return tiers
        j += 1
