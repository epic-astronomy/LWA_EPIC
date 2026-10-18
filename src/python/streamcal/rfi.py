"""Interference detection, subspace estimation and projected calibration for the streamcal daemon.

From the prototype in epic_notes/rfi_mitigation (rfi.py, proto_cal.py), which characterises the
station-fixed, 120 Hz-modulated broadband interferer coupled at the ARX boards (2026-10-17/18) and
measures this method on the 75.8 MHz tap tables: chi^2/dof 525 -> 5.9 with three components.

Notation. R is the measured covariance of one channel and polarisation (cable model applied),
D = diag(R). Whitened quantities carry a subscript w: X_w = D^-1/2 X D^-1/2.

The interference is modelled as a low-rank term that is not tied to any sky direction,

    R = G M G^H + V V^H + Sigma,        V: N x k,  Sigma diagonal (receiver noise),

and is removed by an oblique projector built in the whitened domain,

    P = D^1/2 (I - U U^H) D^-1/2,       U: whitened interference subspace (orthonormal columns).

Calibration minimises || W o P (R - G M G^H - Sigma) P^H ||_F^2 by a majorise-minimise (EM-like)
loop around the unchanged StEFCal/stefcal_sky: the data inside span(U) are replaced by the model,

    R_fill = R_mod + P (R - R_mod) P^H,    R_mod = G M G^H + Sigma,

and StEFCal is run on R_fill. Since P (R_fill - R_mod) P^H = P (R - R_mod) P^H and a projector does
not increase the Frobenius norm, each StEFCal step cannot increase the projected cost.
"""
from __future__ import annotations

import numpy as np


def mp_edge(n, T):
    """Upper Marchenko-Pastur edge of the eigenvalues of a whitened sample covariance (white noise)."""
    return (1.0 + np.sqrt(n / T)) ** 2


def offdiag_noise_edge(n, T):
    """Spectral-norm edge of the off-diagonal sampling noise of a whitened covariance (semicircle)."""
    return 2.0 * np.sqrt(n / T)


def lowrank_offdiag(Xw, kmax=6, niter=6):
    """Eigen-decomposition of a Hermitian matrix whose diagonal is unknown (iterative imputation).

    The diagonal is replaced by that of the rank-kmax approximation, repeated niter times
    (minimum-residual factor analysis). Returns eigenvalues (descending) and eigenvectors."""
    X = np.array(Xw, dtype=np.complex128, copy=True)
    np.fill_diagonal(X, 0.0)
    for _ in range(niter):
        w, v = np.linalg.eigh(X)
        w, v = w[::-1], v[:, ::-1]
        wp = np.clip(w[:kmax], 0, None)
        np.fill_diagonal(X, np.einsum("ak,k,ak->a", v[:, :kmax], wp, v[:, :kmax].conj()).real)
    w, v = np.linalg.eigh(X)
    return w[::-1], v[:, ::-1]


def detect(Ew, T, thresh, kmax=4, rel=None):
    """Interference components in a whitened residual Ew = D^-1/2 (R - G M G^H) D^-1/2 (diagonal ignored).

    A component is declared when its eigenvalue (after diagonal imputation) exceeds the threshold
      t = max(thresh, 3 x 2 sqrt(N/T), rel x median(eigenvalues 5..10)),
    i.e. it must stand above the sampling-noise edge and, with `rel`, above the model-error floor
    measured in the same residual (the bulk of the residual spectrum at full tap rate is model
    error, not noise). Returns (k, eigenvalues, eigenvectors)."""
    n = Ew.shape[0]
    w, v = lowrank_offdiag(Ew, kmax=max(kmax + 2, 10))
    t = max(thresh, 3 * offdiag_noise_edge(n, T))
    if rel:
        t = max(t, rel * float(np.median(w[4:10])))
    k = int(min(kmax, np.sum(w[:kmax] > t)))
    return k, w, v


class Projector:
    """Oblique projector P = D^1/2 (I - U U^H) D^-1/2 and the interference covariance estimate."""

    def __init__(self, U, dh, lam=None):
        self.U = np.asarray(U, complex).reshape(len(dh), -1)
        self.dh = np.asarray(dh, float)
        self.lam = None if lam is None else np.asarray(lam, float)

    @property
    def k(self):
        return self.U.shape[1]

    def apply(self, X):
        """P X P^H for Hermitian X (O(N^2 k))."""
        if self.k == 0:
            return X
        s = np.outer(self.dh, self.dh)
        Xw = X / s
        U = self.U
        XU = Xw @ U
        UXU = U.conj().T @ XU
        Y = Xw - U @ XU.conj().T - XU @ U.conj().T + U @ UXU @ U.conj().T
        return Y * s

    def interference(self):
        """Estimated interference covariance D^1/2 U diag(lam) U^H D^1/2 (lam: whitened powers)."""
        if self.k == 0 or self.lam is None:
            return np.zeros((len(self.dh), len(self.dh)), complex)
        Iw = (self.U * self.lam[None, :]) @ self.U.conj().T
        return Iw * np.outer(self.dh, self.dh)


def model_cov(g, Mfit, R, Iest=None):
    """R_mod = G M G^H with the diagonal set to diag(R) minus the interference estimate
    (the diagonal holds receiver noise that the sky model does not)."""
    Rm = np.outer(g, g.conj()) * Mfit
    d = np.real(np.diag(R)).copy()
    if Iest is not None:
        d = d - np.real(np.diag(Iest))
    np.fill_diagonal(Rm, np.maximum(d, 0.05 * np.real(np.diag(R))))
    return Rm


def chi2_dof(res, R, W, T, ok):
    """Daemon statistic: median over stands of the per-stand normalised residual on W."""
    d = np.maximum(np.real(np.diag(R)), 1e-30)
    chi_ant = (np.abs(res) ** 2 / (np.outer(d, d) / T) * W).sum(1) / np.maximum(W.sum(1), 1)
    return float(np.median(chi_ant[ok])) if ok.any() else float("nan"), chi_ant


_GRID = {}


def sky_focus(u, r, f, n=41):
    """Largest fraction of a component's power that one sky direction captures:
    max_s |a(s)^H u|^2 / (|a|^2 |u|^2), a(s) = exp(2 pi i f r.s / c) on the visible hemisphere.

    A compact sky source gives a value near 1 (its eigenvector follows the array geometry); the
    LWA-SV interferer, which enters downstream of the antennas, gives about 0.05. The diffuse
    sky spreads over many components, each with low eigenvalue."""
    key = (n, r.shape[0])
    if key not in _GRID:
        l = np.linspace(-1, 1, n)
        L, M = np.meshgrid(l, l)
        vis = L ** 2 + M ** 2 < 1
        _GRID[key] = np.c_[L[vis], M[vis], np.sqrt(1 - L[vis] ** 2 - M[vis] ** 2)]
    e = _GRID[key]
    A = np.exp(2j * np.pi * f / 299792458.0 * (r @ e.T))  # [n_ant, pix]
    u = np.asarray(u) / np.linalg.norm(u)
    return float(np.max(np.abs(A.conj().T @ u) ** 2) / r.shape[0])


def select_interference(w, v, r, f, thresh, kmax, max_focus=0.3):
    """Indices of components that are strong (eigenvalue > thresh) and not sky-like
    (sky_focus < max_focus), strongest first, at most kmax."""
    out = []
    for j in range(min(len(w), kmax + 4)):
        if w[j] <= thresh or len(out) >= kmax:
            break
        if sky_focus(v[:, j], r, f) < max_focus:
            out.append(j)
    return out
