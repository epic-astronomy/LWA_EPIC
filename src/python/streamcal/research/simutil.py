import numpy as np

def wishart(R, T, rng):
    """Sample covariance of T complex Gaussian samples with covariance R (Bartlett decomposition)."""
    n = R.shape[0]
    L = np.linalg.cholesky(R + 1e-9 * np.trace(R).real / n * np.eye(n))
    A = np.tril((rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2), -1)
    A[np.diag_indices(n)] = np.sqrt(rng.gamma(T - np.arange(n), 1.0))
    LA = L @ A
    return LA @ LA.conj().T / T

def gain_error(g_est, g_true):
    """rms |g_est/g_true - 1| after removing the common phase (the gauge)."""
    r = g_est / g_true
    r = r * np.conj(np.sum(r)) / abs(np.sum(r))
    return np.sqrt(np.mean(np.abs(r - 1) ** 2)), np.degrees(np.sqrt(np.mean(np.angle(r) ** 2)))


def gain_error_tiptilt(g_est, g_true, r, f):
    """gain_error after also removing a linear phase gradient across the array (an image shift).

    Returns (amp+phase rms, phase rms deg, equivalent image shift in deg)."""
    q = g_est / g_true
    q = q * np.conj(np.sum(q)) / abs(np.sum(q))
    ph = np.angle(q)
    A = np.c_[np.ones(len(ph)), 2 * np.pi * f / 299792458.0 * r[:, :2]]
    c = np.linalg.lstsq(A, ph, rcond=None)[0]
    q2 = q * np.exp(-1j * (A @ c))
    q2 = q2 * np.conj(np.sum(q2)) / abs(np.sum(q2))
    return np.sqrt(np.mean(np.abs(q2 - 1) ** 2)), np.degrees(np.sqrt(np.mean(np.angle(q2) ** 2))), np.degrees(np.hypot(*c[1:]))
