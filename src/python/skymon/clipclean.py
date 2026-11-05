"""
Multi-scale clipping of interference before the search, after CHIME/FRB L1.

The monitor holds a block of images (up to `window`) before searching them.
For every pixel and every sub-band (a group of channels) the block is
standardised with the pixel's running mean and variance at the block's start,
averaged over the sub-band's channels and over 1, 4, 16 and 64 images,
and scaled to unit variance for white noise. At each scale and in each
sub-band the noise is measured from the data (1.4826 times the median absolute
deviation over the sky and the block), and every value beyond `sigma` times
it marks its images in that sub-band and pixel. Each pixel's sub-band is
judged by its departures from its own straight-line trend over the block, so
that a pixel whose mean the running statistics do not follow (a bright source
rising through the beam) is not clipped for its slope; a marked span is set
back to the trend in every channel (kernels.svd_apply), which removes the
excursion and keeps the channel noise, and leaves no edge for the search.

A dispersed burst of S/N s spread over S sub-bands has about s / sqrt(S) in
each: with 8 sub-bands and sigma 6, bursts up to S/N 17 pass untouched and
brighter ones may be clipped; this path is for faint bursts, bright ones are
left to other detection. On live data at 41.8 ms (2026-11-05) the sub-band
means over 4 s varied 36 times more than white noise, in blocks of 0.6 MHz and
5-30 s that the search took for bursts.
"""
from __future__ import annotations

import time

import numpy as np

SCALES = (1, 4, 16, 64)  # in images; the trend takes the longest


def block_means(z: np.ndarray, s: int) -> tuple[np.ndarray, np.ndarray]:
    """Means of z [S][W][P] over consecutive groups of s images (the last may be
    shorter), times sqrt(count): [S][nb][P], and the group starts"""
    W = z.shape[1]
    starts = np.arange(0, W, s)
    cnt = np.diff(np.append(starts, W)).astype(np.float32)
    m = np.add.reduceat(z, starts, axis=1) / np.sqrt(cnt)[None, :, None]
    return m, starts


def clip_block(raw: np.ndarray, mu: np.ndarray, var: np.ndarray, groups, sigma: float = 6.0,
               scales=SCALES, nthreads: int = 1, usable: np.ndarray | None = None,
               sample: int = 7) -> dict:
    """
    Clip a block raw [W][C][P] in place. mu, var [C][P]: running statistics at the
    block's start; groups: (first, end) channel ranges of the sub-bands; usable [P]:
    pixels that take part (past warm-up). Returns the fraction of (sub-band, image,
    pixel) cells clipped at each scale.
    """
    from . import kernels

    W, C, P = raw.shape
    good = np.isfinite(mu) & np.isfinite(var) & (var > 0)
    if usable is not None:
        good &= usable[None, :]
    good_pix = good.all(axis=0)
    good &= good_pix[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.where(good, np.sqrt(np.where(good, var, 1)), 0).astype(np.float32)
        isd = np.where(good, 1.0 / np.where(good, sd, 1), 0).astype(np.float32)
    mu0 = np.where(good, mu, 0).astype(np.float32)
    gstart = np.array([g[0] for g in groups] + [groups[-1][1]], np.int32)
    zsub = kernels.svd_submeans(raw, mu0, isd, gstart, nthreads)  # S, W, P: channel means of z
    zsub[~np.isfinite(zsub)] = 0
    zsub[:, :, ~good_pix] = 0
    nch = np.diff(gstart).astype(np.float32)
    z1 = zsub * np.sqrt(nch)[:, None, None]  # unit variance for white noise
    # each pixel's own straight-line trend over the block: a pixel whose mean the
    # running statistics do not follow (a bright source rising through the beam)
    # is judged by its departures from the trend, and a clipped span is set back
    # to the trend, so that no edge is left for the search
    # robust: a Theil-Sen line (the median of the pairwise slopes, then the median
    # intercept) through the medians of the block's eighths, which an excursion
    # over up to two of them cannot pull
    q = 8
    L = W // q
    tt = (np.arange(W, dtype=np.float32) - (W - 1) / 2.0)
    tq = tt[: q * L].reshape(q, L).mean(1)
    # every 4th image of each segment is enough for its median
    mq = np.median(z1[:, : q * L].reshape(z1.shape[0], q, L, P)[:, :, ::4], axis=2)  # S, q, P
    i, j = np.triu_indices(q, 1)
    slope = np.median((mq[:, j] - mq[:, i]) / (tq[j] - tq[i])[None, :, None], axis=1)  # S, P
    icpt = np.median(mq - slope[:, None, :] * tq[None, :, None], axis=1)
    trend = icpt[:, None, :] + slope[:, None, :] * tt[None, :, None]
    z1 -= trend
    mask = np.zeros(zsub.shape, bool)
    live = np.nonzero(good_pix)[0][::max(1, sample)]
    frac = {}
    if len(live) == 0:
        return dict(frac=frac, clipped=0.0)
    for s in scales:
        if s > W:
            break
        rs = None
        # windows of s images, and the same shifted by half, so that an excursion
        # across a window edge is caught whole by one of them
        for off in ((0, s // 2) if s > 1 else (0,)):
            m, starts = block_means(z1[:, off:], s)
            if rs is None:  # the noise of this scale, from the unshifted windows
                ms = m[:, :, live]
                med = np.median(ms, axis=(1, 2), keepdims=True)
                rs = 1.4826 * np.median(np.abs(ms - med), axis=(1, 2), keepdims=True)
            bad = np.abs(m - med) > sigma * np.maximum(rs, 1e-6)
            bad[:, :, ~good_pix] = False
            if off == 0:
                frac[s] = float(bad[:, :, live].mean())
            if bad.any():
                mask[:, off:] |= np.repeat(bad, np.diff(np.append(starts, W - off)), axis=1)
    if mask.any():
        # the departure from the trend, in the channel-mean units of zsub
        corr = np.where(mask, z1 / np.sqrt(nch)[:, None, None], 0).astype(np.float32)
        kernels.svd_apply(raw, sd, corr, gstart, nthreads)  # the sub-band set back to its trend
    return dict(frac=frac, clipped=float(mask[:, :, live].mean()))


class BlockClipper:
    """The monitor's side of the clipping, with the interface of svdclean.SvdCleaner"""

    def __init__(self, nchan: int, nsub: int = 8, window: int = 256, sigma: float = 6.0,
                 nthreads: int = 1):
        self.nsub = max(1, min(nsub, nchan))
        edges = np.linspace(0, nchan, self.nsub + 1).round().astype(int)
        self.groups = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]
        self.window, self.sigma, self.nthreads = window, sigma, nthreads
        self.blocks = 0
        self.images = 0
        self.seconds = 0.0
        self.last: dict = {}
        self.clipped_sum = 0.0

    def clean(self, raw, mu, var, usable=None) -> None:
        t0 = time.perf_counter()
        self.last = clip_block(raw, mu, var, self.groups, self.sigma, nthreads=self.nthreads,
                               usable=usable)
        self.blocks += 1
        self.images += len(raw)
        self.clipped_sum += self.last.get("clipped", 0.0)
        self.seconds += time.perf_counter() - t0

    def status(self) -> dict:
        return dict(kind="clip", blocks=self.blocks, images=self.images, window=self.window,
                    sigma=self.sigma, last=self.last,
                    clipped_mean=round(self.clipped_sum / self.blocks, 5) if self.blocks else None,
                    ms_per_image=round(1e3 * self.seconds / self.images, 2) if self.images else None)
