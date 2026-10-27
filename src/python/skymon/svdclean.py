"""
Removal of interference shared by many pixels: the Marchenko-Pastur cut.

Interference seen through the sidelobes (and gain steps, slow common drifts,
the scintillation of bright sources leaking over the image) changes many
pixels together. In the pixels' standardised time series Z (T images x P
pixels) it appears as eigenvalues of Z Z^T / P above what noise alone allows:
the upper edge of the Marchenko-Pastur law, sigma^2 (1 + sqrt(c))^2 for
c = T / P. A burst occupies a beam, a few pixels of P, and adds about
amplitude^2 / P to an eigenvalue: it stays below the edge.

Neighbouring pixels share the beam and the dirty image's sidelobes spread
noise over the sky, so the noise follows the law for fewer, effectively
independent pixels: sigma^2 and c are fitted to the lower 70% of the
eigenvalues, and the components above 1.05 times the fitted edge are removed.

Live, the monitor delays its search by a block of images (up to `window`; it
cleans and releases the block at every update of the tracked pixels, every
10 s, so that a block never spans two pixel sets, and before a calibration
hold): the components above the edge are found in the sub-band averages of
the standardised block and removed from every channel of the same block
(clean_block). A flash is only removed by components that include it:
interference fixed to the ground drifts across sky-fixed pixels, so patterns
learned from earlier images do not match new flashes (patterns from the
previous 43 s raised the false triggers by 10% on the 2026-10-26 recording;
blocks of 256 images lowered them from 21,695 to 502).

Bright bursts (S/N in the hundreds) can cross the edge and be removed: this
path is for faint bursts, and bright ones are left to other detection.

On 10 minutes of 192-channel data at 75-80 MHz (2026-10-26, offline,
43 s windows): false triggers 37,178 -> 39, sky-wide flash images to zero,
injected bursts of S/N 8.5-85 kept 97-144% of their S/N.
"""
from __future__ import annotations

import time

import numpy as np


def mp_quantiles(sig2: float, c: float, q: np.ndarray) -> np.ndarray:
    """Quantiles q of the Marchenko-Pastur law (ratio c <= 1, variance sig2), numerically"""
    a, b = sig2 * (1 - np.sqrt(c)) ** 2, sig2 * (1 + np.sqrt(c)) ** 2
    x = np.linspace(a, b, 4000)[1:]
    pdf = np.sqrt(np.clip((b - x) * (x - a), 0, None)) / (2 * np.pi * sig2 * c * x)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    return np.interp(q, cdf, x)


def fit_bulk(ev: np.ndarray, frac: float = 0.7) -> tuple[float, float]:
    """sigma^2 and c of the Marchenko-Pastur law matching the lower frac of the eigenvalues"""
    ev = np.sort(ev)
    q = (np.arange(int(frac * len(ev))) + 0.5) / len(ev)
    emp = ev[: len(q)]
    best = None
    for c in np.geomspace(0.02, 1.0, 60):
        base = mp_quantiles(1.0, c, q)
        s2 = float(np.dot(base, emp) / np.dot(base, base))
        err = float(np.mean((s2 * base - emp) ** 2))
        if best is None or err < best[0]:
            best = (err, s2, c)
    return best[1], best[2]


def components(z: np.ndarray, margin: float = 1.05, max_k: int = 64) -> tuple[np.ndarray, dict]:
    """Sky patterns (P x k, orthonormal columns) of the components of z (T x P) above the edge"""
    T, P = z.shape
    g = (z @ z.T) / P
    ev, u = np.linalg.eigh(g.astype(np.float64))
    ev, u = ev[::-1], u[:, ::-1]
    s2, c = fit_bulk(ev)
    edge = s2 * (1 + np.sqrt(c)) ** 2 * margin
    k = int(min(max_k, np.sum(ev > edge)))
    info = dict(k=k, sigma2=s2, c_eff=c, edge=edge, top=ev[:5].round(2).tolist())
    if k == 0:
        return np.zeros((P, 0), np.float32), info
    # V = Z^T U S^-1 with S = sqrt(P ev): the right singular vectors of z, orthonormal
    # as they are (eigenvectors of Z^T Z)
    v = (z.T @ u[:, :k].astype(np.float32)) / np.sqrt(P * ev[:k]).astype(np.float32)
    return v.astype(np.float32), info


def clean_block(raw: np.ndarray, mu: np.ndarray, var: np.ndarray, groups, margin: float = 1.05,
                max_k: int = 64, nthreads: int = 1) -> list[dict]:
    """
    Clean a block of images in place with the components of the block itself.
    raw: [W][nchan][P] float32; mu, var: [nchan][P] (running statistics at the block's
    start); groups: (first, end) channel ranges of the sub-bands. The components are
    found in the standardised sub-band means of the block; the part of each sub-band
    mean they account for is removed from every channel of the sub-band, in the
    channel's own units (interference broad over a sub-band of a few hundred kHz;
    narrowband interference is classified separately). The element-wise passes run in
    the compiled kernels (kernels.svd_submeans, svd_apply), the rest in BLAS.
    A flash is only removed by components that include it: interference fixed to the
    ground drifts across sky-fixed pixels, so patterns learned from earlier images do
    not match new flashes (on the 2026-10-26 recording, patterns from the previous
    43 s raised the false triggers by 10%).
    """
    from . import kernels

    W, C, P = raw.shape
    sd = np.sqrt(np.maximum(var, 0)).astype(np.float32)
    isd = np.where(sd > 0, 1.0 / np.where(sd > 0, sd, 1), 0).astype(np.float32)
    gstart = np.array([g[0] for g in groups] + [groups[-1][1]], np.int32)
    zsub = kernels.svd_submeans(raw, mu, isd, gstart, nthreads)  # S, W, P
    corr = np.zeros_like(zsub)
    infos = []
    any_k = False
    for s in range(len(groups)):
        zs = zsub[s]
        live = np.any(zs != 0, axis=0)
        info = dict(k=0)
        if live.sum() > 2 * W:
            zl = zs[:, live]
            v, info = components(zl - zl.mean(0), margin, max_k)
            if v.shape[1]:
                corr[s][:, live] = (zl @ v) @ v.T
                any_k = True
        infos.append(info)
    if any_k:
        kernels.svd_apply(raw, sd, corr, gstart, nthreads)
    return infos


class SvdCleaner:
    """The monitor's side of the cut: the sub-band groups, the parameters and the last results"""

    def __init__(self, nchan: int, nsub: int = 8, window: int = 256, margin: float = 1.05,
                 max_k: int = 64, nthreads: int = 1):
        self.nsub = max(1, min(nsub, nchan))
        edges = np.linspace(0, nchan, self.nsub + 1).round().astype(int)
        self.groups = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]
        self.window, self.margin, self.max_k = window, margin, max_k
        self.nthreads = nthreads
        self.blocks = 0
        self.images = 0
        self.last: list[dict] = []
        self.seconds = 0.0

    def clean(self, raw: np.ndarray, mu: np.ndarray, var: np.ndarray) -> None:
        """Clean a block raw [W][nchan][P] in place (clean_block)"""
        t0 = time.perf_counter()
        self.last = clean_block(raw, mu, var, self.groups, self.margin, self.max_k, self.nthreads)
        self.blocks += 1
        self.images += len(raw)
        self.seconds += time.perf_counter() - t0

    def status(self) -> dict:
        return dict(blocks=self.blocks, images=self.images, window=self.window,
                    k=[i.get("k", 0) for i in self.last],
                    edge=[round(i["edge"], 3) for i in self.last if "edge" in i],
                    ms_per_image=round(1e3 * self.seconds / self.images, 2) if self.images else None)
