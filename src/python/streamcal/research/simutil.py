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
