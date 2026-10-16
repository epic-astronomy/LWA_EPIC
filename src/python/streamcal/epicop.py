"""Exact CPU model of EPIC (epic++ phoenix) image formation for one channel.

The accumulated MOFF image is a linear function of the gain-weighted
visibility matrix W R W^H (Thyagarajan et al. 2017, eq. 11-13). With the
epic++ gridder (5x5 cells per stand, anchored at trunc(pos + d)) the Fourier
transform of the accumulated image is

    D(u) = sum_{a,b} w_a w_b^* R_ab  Q_ab(u - (c_a - c_b))        (mod N grid)

where c_a is stand a's anchor cell and Q_ab is the 9x9 cross-correlation of
the two stands' 5x5 kernels. This module builds c_a, the kernels and the
Q_ab patches, and evaluates D (forward, "scatter") and its adjoint
("gather", which degrids a uv-plane array back onto baselines). It is the
operator the grid-domain calibrator inverts.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import numba as nb

from .geometry import CHAN_BW, SOL, eg  # noqa: F401


@dataclass
class EpicConfig:
    grid_size: int = 128
    grid_res: float = 1.056  # deg (production --imageres)
    support: int = 5
    wproj: bool = True

    @property
    def delta(self):
        """uv cell size in wavelengths."""
        return 1.0 / (2 * self.grid_size * np.sin(np.radians(self.grid_res) / 2))


def stand_frame_positions():
    """Stand positions in the grid frame (rows x, y, w) in metres, and the used mask."""
    _, F, _ = eg.array_frame()
    xyz = eg.stand_xyz()
    return xyz @ F.T, eg.used_stands()


def grid_positions(cfg: EpicConfig, freq_hz: float):
    """Float32 grid positions exactly as MOFFCorrelator::ResetAntpos builds them."""
    pf, used = stand_frame_positions()
    loc = pf.copy()
    loc[[i for i, a in enumerate(eg.lwasv.antennas[::2]) if a.stand.id == 256], :] = 0.0
    loc = loc / cfg.delta
    loc -= loc.min(axis=0, keepdims=True)
    loc -= loc.max(axis=0, keepdims=True) / 2.0
    lam = SOL / freq_hz
    pos = (cfg.grid_size / 2 + loc / lam).astype(np.float32)
    return pos, used


def _prolate_profile(n=2048):
    from scipy.special import pro_ang1, pro_cv

    c = 5.356 * np.pi / 2.0
    u = np.linspace(0, 1, n, endpoint=False)
    s = pro_ang1(0, 0, c, u)[0] * np.sqrt(1 - u * u)
    s[-1] = 0
    return u, s


_PROF = None


def _gcf(r):
    """Separable prolate kernel factor at |r| in units of the kernel half-width."""
    global _PROF
    if _PROF is None:
        _PROF = _prolate_profile()
    u, s = _PROF
    return np.interp(np.abs(r), u, s, right=0.0)


def real_kernels(cfg: EpicConfig, pos: np.ndarray, freq_hz: float):
    """ComputeGcfElements: 5x5 pixel-integrated prolate kernel per stand, sum 1.

    Returns (kern[nstand, S*S] float, xp[nstand,S*S] int, yp[nstand,S*S] int).
    """
    S = cfg.support
    half = S // 2
    e = np.arange(S * S)
    dy = (half - e // S).astype(np.float32)
    dx = (e % S - half).astype(np.float32)
    xp = np.trunc(pos[:, 0:1] + dx).astype(np.int64)
    yp = np.trunc(pos[:, 1:2] + dy).astype(np.int64)
    lam = SOL / freq_hz
    # kernel half-width sqrt(Aeff) = sqrt(0.1313) lambda, in cell units
    hw = np.sqrt(0.1313) * lam / (lam * cfg.delta)
    sub = (np.arange(5) + 0.5) / 5.0
    ox = (xp - pos[:, 0:1])[..., None, None] + sub[None, None, :, None]
    oy = (yp - pos[:, 1:2])[..., None, None] + sub[None, None, None, :]
    k = (_gcf(ox / hw) * _gcf(oy / hw)).sum(axis=(-1, -2))
    k /= np.maximum(k.sum(axis=1, keepdims=True), 1e-30)
    return k.astype(np.float32), xp, yp


def kernels(cfg: EpicConfig, chan: int):
    """Complex per-stand kernels (w-projected if cfg.wproj), anchors and used mask for one channel."""
    f = chan * CHAN_BW
    pos, used = grid_positions(cfg, f)
    kr, xp, yp = real_kernels(cfg, pos, f)
    if cfg.wproj:
        kc = eg.wproj_kernel_weights(kr[None], pos[None], chan, 1, cfg.support, cfg.grid_size, cfg.grid_res)[0]
    else:
        kc = kr.astype(np.complex64)
    return kc, xp, yp, pos, used


@dataclass
class GridOp:
    """Precomputed operator for one channel: baselines (a,b) a != b over used stands."""

    N: int  # grid size
    ant: np.ndarray  # used stand indices (into 256)
    a: np.ndarray  # baseline first index (into ant)
    b: np.ndarray
    off: np.ndarray  # [nbl, 2] integer anchor offset (dy, dx) of Q_ab's centre
    Q: np.ndarray  # [nbl, 9, 9] complex64
    Qn2: np.ndarray  # [nbl] ||Q_ab||^2
    autoQ: np.ndarray  # [nant, 9, 9] Q_aa (autocorrelation footprint)
    idx: np.ndarray = None  # [nbl, 81] flat grid cell of each Q element
    Qf: np.ndarray = None  # [nbl, 81] Q flattened, complex128

    def __post_init__(self):
        S2 = self.Q.shape[1]
        h = S2 // 2
        v = np.arange(S2) - h
        yy = (self.off[:, 0, None, None] + v[None, :, None]) % self.N
        xx = (self.off[:, 1, None, None] + v[None, None, :]) % self.N
        self.idx = (yy * self.N + xx).reshape(len(self.off), -1).astype(np.int32)
        self.Qf = self.Q.reshape(len(self.off), -1).astype(np.complex128)


def build_gridop(cfg: EpicConfig, chan: int, min_uv_cells: float = 0.0) -> GridOp:
    kc, xp, yp, pos, used = kernels(cfg, chan)
    S = cfg.support
    ant = np.flatnonzero(used)
    # 5x5 patch of each stand, indexed [iy, ix] with iy along +y (row 0 = lowest y)
    P = np.zeros((len(ant), S, S), np.complex64)
    y0 = np.zeros(len(ant), np.int64)
    x0 = np.zeros(len(ant), np.int64)
    for i, s in enumerate(ant):
        y0[i] = yp[s].min()
        x0[i] = xp[s].min()
        P[i, yp[s] - y0[i], xp[s] - x0[i]] = kc[s]
    n = len(ant)
    ia, ib = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    m = ia != ib
    a, b = ia[m], ib[m]
    # Q_ab(v) = sum_r K_a(r + v) K_b^*(r), v in [-(S-1), S-1]^2, relative to (c_a - c_b)
    Fa = np.fft.fft2(P, s=(2 * S - 1, 2 * S - 1))
    Q = np.fft.ifft2(Fa[a] * np.conj(Fa[b]))  # circular xcorr, index v mod (2S-1)
    Q = np.fft.fftshift(Q, axes=(-2, -1)).astype(np.complex64)
    off = np.stack([y0[a] - y0[b], x0[a] - x0[b]], axis=1)
    if min_uv_cells > 0:
        keep = np.hypot(*off.T) >= min_uv_cells
        a, b, Q, off = a[keep], b[keep], Q[keep], off[keep]
    autoQ = np.fft.fftshift(np.fft.ifft2(np.abs(Fa) ** 2), axes=(-2, -1)).astype(np.complex64)
    return GridOp(cfg.grid_size, ant, a, b, off, Q, (np.abs(Q) ** 2).sum(axis=(1, 2)), autoQ)


@nb.njit(parallel=False, fastmath=True, cache=True)
def _scatter(vis, off, Q, N, out):
    nbl = vis.shape[0]
    h = Q.shape[1] // 2
    for k in range(nbl):
        v = vis[k]
        if v == 0:
            continue
        oy = off[k, 0]
        ox = off[k, 1]
        for i in range(Q.shape[1]):
            yy = (oy + i - h) % N
            for j in range(Q.shape[2]):
                xx = (ox + j - h) % N
                out[yy, xx] += v * Q[k, i, j]
    return out


@nb.njit(parallel=False, fastmath=True, cache=True)
def _gather(D, off, Q, N, out):
    nbl = out.shape[0]
    h = Q.shape[1] // 2
    for k in range(nbl):
        oy = off[k, 0]
        ox = off[k, 1]
        acc = 0j
        for i in range(Q.shape[1]):
            yy = (oy + i - h) % N
            for j in range(Q.shape[2]):
                xx = (ox + j - h) % N
                acc += np.conj(Q[k, i, j]) * D[yy, xx]
        out[k] = acc
    return out


@nb.njit(fastmath=True, cache=True)
def _scatter_f(vis, idx, Qf, out):
    for k in range(vis.shape[0]):
        v = vis[k]
        if v == 0:
            continue
        for e in range(idx.shape[1]):
            out[idx[k, e]] += v * Qf[k, e]
    return out


@nb.njit(parallel=True, fastmath=True, cache=True)
def _gather_f(D, idx, Qf, out):
    for k in nb.prange(out.shape[0]):
        acc = 0j
        for e in range(idx.shape[1]):
            acc += np.conj(Qf[k, e]) * D[idx[k, e]]
        out[k] = acc
    return out


def scatter(op: GridOp, vis: np.ndarray) -> np.ndarray:
    """D = sum_bl vis_bl Q_bl placed at the baseline offset (no autocorrelations)."""
    out = np.zeros(op.N * op.N, np.complex128)
    return _scatter_f(np.ascontiguousarray(vis, np.complex128), op.idx, op.Qf, out).reshape(op.N, op.N)


def gather(op: GridOp, D: np.ndarray) -> np.ndarray:
    """Adjoint of scatter: rho_bl = <Q_bl, D>."""
    out = np.zeros(len(op.a), np.complex128)
    return _gather_f(np.ascontiguousarray(D, np.complex128).ravel(), op.idx, op.Qf, out)


def autos(op: GridOp, p: np.ndarray) -> np.ndarray:
    """Autocorrelation pedestal sum_a p_a Q_aa at the origin."""
    S2 = op.autoQ.shape[1]
    h = S2 // 2
    out = np.zeros((op.N, op.N), np.complex128)
    tot = np.tensordot(p, op.autoQ, axes=(0, 0))
    for i in range(S2):
        for j in range(S2):
            out[(i - h) % op.N, (j - h) % op.N] += tot[i, j]
    return out


def image_from_vis(op: GridOp, wR: np.ndarray, with_autos=True) -> np.ndarray:
    """Accumulated EPIC image (unnormalised, before gridding correction) from W R W^H on used stands."""
    D = scatter(op, wR[op.a, op.b])
    if with_autos:
        D += autos(op, np.real(np.diag(wR)))
    # D(u) = sum_r G(r+u) G^*(r) is the FFT of |IFFT(G)|^2 up to N^2 scale
    return np.real(np.fft.ifft2(D)) * op.N ** 2


def moff_image(op: GridOp, cfg: EpicConfig, chan: int, w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Literal MOFF on voltages X[nant_used, T] for checking image_from_vis: grid, IFFT, square, sum."""
    kc, xp, yp, pos, used = kernels(cfg, chan)
    N = cfg.grid_size
    img = np.zeros((N, N))
    for t in range(X.shape[1]):
        G = np.zeros((N, N), np.complex128)
        for i, s in enumerate(op.ant):
            np.add.at(G, (yp[s], xp[s]), kc[s] * w[i] * X[i, t])
        img += np.abs(np.fft.ifft2(G) * N * N) ** 2
    return img


def density(op: GridOp) -> np.ndarray:
    """Per-cell noise variance (equal-power antennas): sum_bl |Q_bl(u)|^2."""
    out = np.zeros(op.N * op.N)
    np.add.at(out, op.idx.ravel(), (np.abs(op.Qf) ** 2).ravel())
    return out.reshape(op.N, op.N)
