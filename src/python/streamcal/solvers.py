"""CPU gain solvers for EPIC and a tracking filter (prototype: research/sim_loop.py).

stefcal      visibility-domain StEFCal (Salvini & Wijnholds 2014) for the
             voltage-tap path: R ~ G M G^H on a baseline mask.
gridcal      grid-domain solver for the image path: fits the residual gains
             delta to the Fourier transform of accumulated EPIC images,
             D ~ G(phi delta M delta^H phi^H), by Gauss-Newton with a
             matrix-free, Jacobi-preconditioned conjugate gradient. The
             forward operator is epicop.scatter, its adjoint epicop.gather.
Tracker      per-antenna Kalman filter on (phase offset, delay) plus a slow
             per-channel complex bandpass, fed by per-channel solutions.

Gauge: an overall phase per polarisation is unobservable; every solution is
referenced so that sum_a g_a has zero phase.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from . import epicop as E  # grid-domain solver only; needs numba
except ImportError:  # pragma: no cover
    E = None


def reference_phase(g):
    s = np.sum(g)
    return g * (np.conj(s) / abs(s)) if abs(s) > 0 else g


# ---------------------------------------------------------------- StEFCal


def stefcal(R, M, W, g0=None, niter=60, tol=1e-6, free=None):
    """Solve R ~ diag(g) (M + sum_k s_k F_k) diag(g)^H on weights W (0 on autos / short baselines).

    free: optional list of model matrices F_k whose real scale s_k is solved too
          (the Sun). Returns (g, s, n_iter).
    """
    n = R.shape[0]
    g = np.ones(n, complex) if g0 is None else g0.copy()
    s = np.zeros(0) if not free else np.ones(len(free))
    Mt = M + sum(sk * F for sk, F in zip(s, free or []))
    for it in range(niter):
        g_old = g
        Z = (g[:, None] * Mt) * W
        num = np.sum(np.conj(R * W) * Z, axis=0)
        den = np.sum(np.abs(Z) ** 2, axis=0)
        gn = np.where(den > 0, num / np.where(den > 0, den, 1), g)
        g = 0.5 * (gn + g_old) if it % 2 == 1 else gn
        if free and it % 4 == 3:
            GG = np.outer(g, g.conj()) * W
            A = np.stack([(GG * F).ravel() for F in free], 1)
            b = (R * W - GG * M).ravel()
            s = np.linalg.lstsq(np.vstack([A.real, A.imag]), np.concatenate([b.real, b.imag]), rcond=None)[0]
            Mt = M + sum(sk * F for sk, F in zip(s, free))
        if it % 2 == 1 and np.linalg.norm(g - g_old) / np.linalg.norm(g) < tol:
            break
    return reference_phase(g), s, it + 1


def baseline_mask(op: E.GridOp, n, min_cells):
    W = np.zeros((n, n))
    keep = np.hypot(*op.off.T) >= min_cells
    W[op.a[keep], op.b[keep]] = 1
    return W


# ---------------------------------------------------------------- grid-domain


def uv_mask(N, min_cells, margin=4.5):
    k = np.fft.fftfreq(N) * N
    r = np.hypot(k[:, None], k[None, :])
    return (r >= min_cells + margin).astype(float)


@dataclass
class GridProblem:
    op: E.GridOp
    Vt: np.ndarray  # [nbl] phi_a phi_b^* M_ab  (model, geometric phase applied)
    mask: np.ndarray  # [N, N] 0/1
    free: list = None  # list of [nbl] arrays phi phi^* F_k (solved real scale)

    def model_vis(self, d, s=None):
        V = self.Vt
        if self.free:
            V = V + sum(sk * F for sk, F in zip(s, self.free))
        return d[self.op.a] * np.conj(d[self.op.b]) * V

    def J(self, d, Vs, x):
        a, b = self.op.a, self.op.b
        v = (x[a] * np.conj(d[b]) + d[a] * np.conj(x[b])) * Vs
        return self.mask * E.scatter(self.op, v)

    def JH(self, d, Vs, r):
        a, b = self.op.a, self.op.b
        rho = E.gather(self.op, self.mask * r)
        out = np.zeros(len(d), complex)
        np.add.at(out, a, np.conj(np.conj(d[b]) * Vs) * rho)
        np.add.at(out, b, d[a] * Vs * np.conj(rho))
        return out

    def precond(self, d, Vs):
        a, b = self.op.a, self.op.b
        w = np.abs(Vs) ** 2 * self.op.Qn2
        p = np.zeros(len(d))
        np.add.at(p, a, np.abs(d[b]) ** 2 * w)
        np.add.at(p, b, np.abs(d[a]) ** 2 * w)
        return p


def _rdot(x, y):
    return float(np.real(np.vdot(x, y)))


def gridcal(prob: GridProblem, D, d0=None, n_outer=3, n_cg=12, damping=1e-3, s0=None):
    """Fit residual gains d to D (the FFT of the accumulated image, autos masked).

    Returns (d, s, info). d multiplies the gains that were applied when the
    image was made: g_new = g_applied * d.
    """
    op = prob.op
    n = len(op.ant)
    d = np.ones(n, complex) if d0 is None else d0.copy()
    s = np.ones(len(prob.free)) if prob.free else None
    if s0 is not None and prob.free:
        s = np.array(s0, float)
    Dm = prob.mask * D
    hist = []
    for it in range(n_outer):
        Vs = prob.Vt if not prob.free else prob.Vt + sum(sk * F for sk, F in zip(s, prob.free))
        r = Dm - prob.mask * E.scatter(op, d[op.a] * np.conj(d[op.b]) * Vs)
        hist.append(np.linalg.norm(r) ** 2)
        g = prob.JH(d, Vs, r)
        P = prob.precond(d, Vs)
        lam = damping * np.median(P)
        Pinv = 1.0 / (P + lam)
        x = np.zeros(n, complex)
        res = g.copy()
        z = Pinv * res
        p = z.copy()
        rz = _rdot(res, z)
        for _ in range(n_cg):
            Ap = prob.JH(d, Vs, prob.J(d, Vs, p)) + lam * p
            alpha = rz / _rdot(p, Ap)
            x += alpha * p
            res -= alpha * Ap
            z = Pinv * res
            rz_new = _rdot(res, z)
            p = z + (rz_new / rz) * p
            rz = rz_new
        d = d + x
        if prob.free:
            base = prob.mask * E.scatter(op, d[op.a] * np.conj(d[op.b]) * prob.Vt)
            A = [prob.mask * E.scatter(op, d[op.a] * np.conj(d[op.b]) * F) for F in prob.free]
            Am = np.stack([x_.ravel() for x_ in A], 1)
            bb = (Dm - base).ravel()
            s = np.linalg.lstsq(np.vstack([Am.real, Am.imag]), np.concatenate([bb.real, bb.imag]), rcond=None)[0]
    Vs = prob.Vt if not prob.free else prob.Vt + sum(sk * F for sk, F in zip(s, prob.free))
    r = Dm - prob.mask * E.scatter(op, d[op.a] * np.conj(d[op.b]) * Vs)
    hist.append(np.linalg.norm(r) ** 2)
    # per-antenna variance estimate: sigma^2 / diag(J^H J)
    ncell = prob.mask.sum()
    sig2 = hist[-1] / max(ncell - 2 * n, 1)
    var = sig2 / np.maximum(prob.precond(d, Vs), 1e-30)
    return reference_phase(d), s, dict(chi2=hist, var=var)


# ---------------------------------------------------------------- tracking


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


class Tracker:
    """Per-antenna Kalman filter on x = (phase offset at nu_c [rad], delay [ns]).

    Measurements are per-channel phases of the absolute gain solutions; the
    amplitude and a slow per-channel phase bandpass are exponentially averaged.
    Produces a gain table for any channel list.
    """

    def __init__(self, n_ant, nu_hz, nu_c, q_phase=np.radians(2.0) ** 2, q_delay=0.05 ** 2,
                 tau_bp=600.0, tau_amp=60.0, jump_nsigma=5.0, jump_count=3):
        self.n = n_ant
        self.nu = np.asarray(nu_hz, float)  # channels being solved
        self.nu_c = nu_c
        self.x = np.zeros((n_ant, 2))
        self.P = np.tile(np.diag([np.pi ** 2, 10.0 ** 2]), (n_ant, 1, 1))
        self.q = np.diag([q_phase, q_delay])  # per second
        self.logamp = np.zeros((n_ant, len(self.nu)))
        self.bp = np.zeros((n_ant, len(self.nu)))  # slow per-channel phase
        self.tau_bp, self.tau_amp = tau_bp, tau_amp
        self.jump_nsigma, self.jump_count = jump_nsigma, jump_count
        self.strikes = np.zeros(n_ant, int)
        self.initialised = False

    def H(self, nu):
        return np.stack([np.ones_like(nu), 2 * np.pi * (nu - self.nu_c) * 1e-9], 1)

    def phase_model(self, nu):
        return self.x @ self.H(nu).T  # [n_ant, nchan]

    def update(self, g, var, dt):
        """g [n_ant, nchan] absolute gain solutions on self.nu, var [n_ant, nchan] |dg|^2 variances."""
        Hm = self.H(self.nu)
        self.P += self.q * dt
        ph = np.angle(g)
        sig2 = np.clip(var / np.maximum(np.abs(g) ** 2, 1e-12), 1e-6, 10.0)  # phase variance
        good = np.isfinite(ph) & (np.abs(g) > 0)
        if not self.initialised:
            self._initialise(g, good)
        for a in range(self.n):
            m = good[a]
            if m.sum() < 2:
                continue
            pred = Hm[m] @ self.x[a] + self.bp[a, m]
            y = wrap(ph[a, m] - pred)
            Ha = Hm[m]
            Rm = np.diag(sig2[a, m])
            S = Ha @ self.P[a] @ Ha.T + Rm
            Si = np.linalg.inv(S)
            nis = y @ Si @ y / m.sum()
            if nis > self.jump_nsigma ** 2:
                self.strikes[a] += 1
                if self.strikes[a] < self.jump_count:
                    continue
                # persistent jump: re-acquire this antenna
                self.P[a] = np.diag([np.pi ** 2, 10.0 ** 2])
                S = Ha @ self.P[a] @ Ha.T + Rm
                Si = np.linalg.inv(S)
            self.strikes[a] = 0
            K = self.P[a] @ Ha.T @ Si
            self.x[a] += K @ y
            self.P[a] = (np.eye(2) - K @ Ha) @ self.P[a]
            self.x[a, 0] = wrap(self.x[a, 0])
            # slow per-channel phase bandpass (zero mean, no slope)
            resid = wrap(ph[a, m] - Hm[m] @ self.x[a])
            resid -= Hm[m] @ np.linalg.lstsq(Hm[m], resid, rcond=None)[0]
            k = min(1.0, dt / self.tau_bp)
            self.bp[a, m] += k * wrap(resid - self.bp[a, m])
        la = np.log(np.maximum(np.abs(np.nan_to_num(g)), 1e-6))
        k = min(1.0, dt / self.tau_amp)
        self.logamp = np.where(good, self.logamp + k * (la - self.logamp), self.logamp)

    def _initialise(self, g, good):
        Hm = self.H(self.nu)
        for a in range(self.n):
            m = good[a]
            if m.sum() < 2:
                continue
            ph = np.unwrap(np.angle(g[a, m]))
            self.x[a] = np.linalg.lstsq(Hm[m], ph, rcond=None)[0]
            self.x[a, 0] = wrap(self.x[a, 0])
            self.P[a] = np.diag([np.radians(10) ** 2, 1.0])
        self.logamp = np.where(good, np.log(np.maximum(np.abs(np.nan_to_num(g)), 1e-6)), 0.0)
        self.initialised = True

    def gains(self, nu=None):
        """Gain table on the solved channels (nu=None) or interpolated to other channels."""
        if nu is None:
            return np.exp(self.logamp + 1j * (self.phase_model(self.nu) + self.bp))
        nu = np.asarray(nu, float)
        la = np.array([np.interp(nu, self.nu, r) for r in self.logamp])
        bp = np.array([np.interp(nu, self.nu, r) for r in self.bp])
        return np.exp(la + 1j * (self.phase_model(nu) + bp))


def ml_weights(op: E.GridOp, min_cells, margin=0.0, floor=1e-3):
    """sqrt of inverse-noise cell weights times the short-spacing mask (pass as GridProblem.mask)."""
    rho = E.density(op)
    w = uv_mask(op.N, min_cells, margin) / np.maximum(rho, floor * rho.max())
    w[rho < floor * rho.max()] = 0
    return np.sqrt(w / w.max())


# ------------------------------------------------- StEFCal with a live sky model


def _tangent(s):
    """Orthonormal tangent basis (t1 ~ east, t2 ~ north) at unit vector s (ENU)."""
    t1 = np.cross([0.0, 0.0, 1.0], s)
    if np.linalg.norm(t1) < 1e-8:
        t1 = np.array([1.0, 0.0, 0.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(s, t1)
    return t1, t2


def shift_direction(s, d):
    """Move unit vector s by d = (d1, d2) radians in its tangent plane."""
    t1, t2 = _tangent(s)
    v = s + d[0] * t1 + d[1] * t2
    return v / np.linalg.norm(v)


def point_matrix(r, f, s, app):
    """app * a a^H with a = exp(2 pi i f r.s / c) (r: stand ENU positions [n, 3])."""
    a = np.exp(2j * np.pi * f / 299792458.0 * (r @ s))
    return app * np.outer(a, a.conj())


def stefcal_sky(R, M_fixed, sources, W, r, f, g0=None, niter=4, n_outer=20, solve_pos=True, fixed_pos=None,
                max_step_deg=1.0, tol=1e-6, max_off_deg=None, scale_bounds=None, prior_off_deg=None):
    """StEFCal with the bright compact sources as live model components.

    R, M_fixed  data and the fixed part of the model (diffuse sky, faint sources)
    sources     list of (enu unit vector, apparent flux in Jy) from the forward model
    r, f        stand ENU positions [n, 3] and frequency (Hz)
    g0          starting gains (the tracker prediction when streaming)
    Solves the gains and, per source, a real flux scale and (solve_pos) a
    tangent-plane offset in radians; fixed_pos [k, 2] imposes offsets instead.

    Each pass first updates the source parameters by linear least squares with
    the current gains (scale and scale x offset enter linearly around the
    current directions), then runs `niter` StEFCal iterations warm-started from
    the current gains. Updating the sky first matters in streaming, where the
    gains are known from the previous interval but the sky (scintillation, a
    solar burst, refraction) may have changed.
    Guard rails (LWA-SV TBF data at 28-47 MHz drove free sources to near-zero flux and
    6-11 deg offsets when the diffuse model was poor): a source whose offset exceeds
    max_off_deg (default 0.5 deg x (74 MHz / f)^2, a generous refraction prior) or whose
    scale leaves scale_bounds[k] (default 0.25-4) is reset to its catalogue flux and
    position and frozen for the rest of the solve. info["frozen"] lists them.
    prior_off_deg: optional Gaussian refraction prior (1-sigma per axis, degrees, at this
    frequency): a Tikhonov row per offset component pulls the total offset toward zero, weighted
    by the data residual rms / prior width. A poorly constrained offset (or one that would absorb
    direction-dependent phase errors or sky-model error) then stays near the catalogue position;
    a well-measured refraction shift still comes through.
    Returns (g, scale[k], offset[k, 2], info).
    """
    K = len(sources)
    if max_off_deg is None:
        max_off_deg = 0.5 * (74e6 / f) ** 2
    if scale_bounds is None:
        scale_bounds = [(0.25, 4.0)] * K
    frozen = np.zeros(K, bool)
    n = R.shape[0]
    scale = np.ones(K)
    off = np.zeros((K, 2)) if fixed_pos is None else np.array(fixed_pos, float).copy()
    k2pi = 2 * np.pi * f / 299792458.0
    it_total = 0
    if g0 is None:  # cold start: gains against the nominal sky before the sky parameters move
        M0 = M_fixed + sum(point_matrix(r, f, src, app) for src, app in sources)
        g, _, it_total = stefcal(R, M0, W, niter=60)
    else:
        g = g0.copy()
    for outer in range(n_outer):
        dirs = [shift_direction(src, off[k]) for k, (src, _) in enumerate(sources)]
        P = [point_matrix(r, f, dirs[k], sources[k][1]) for k in range(K)]
        if K:
            GG = np.outer(g, g.conj()) * W
            cols = []
            for k in range(K):
                cols.append((GG * P[k]).ravel())
                if solve_pos:
                    t1, t2 = _tangent(dirs[k])
                    for t in (t1, t2):
                        b = r @ t
                        cols.append((GG * P[k] * (1j * k2pi) * (b[:, None] - b[None, :])).ravel())
            step = 3 if solve_pos else 1
            live = np.repeat(~frozen, step)
            A = np.stack(cols, 1)
            y = (R * W - GG * M_fixed).ravel()
            for k in np.flatnonzero(frozen):  # frozen sources: catalogue flux, no offset
                y = y - (GG * P[k]).ravel()
            theta = np.zeros(A.shape[1])
            if live.any():
                Al = A[:, live]
                Ar, yr = np.vstack([Al.real, Al.imag]), np.concatenate([y.real, y.imag])
                if solve_pos and prior_off_deg:
                    # rows alpha * (u_t + s_k off_t) = 0 for u_t = s_k d_t, the offset increment
                    # times the scale, so the prior acts on the total offset off + d
                    th0 = np.zeros(A.shape[1])
                    th0[0::step] = scale
                    res0 = yr - np.vstack([A.real, A.imag]) @ th0
                    alpha = np.sqrt(np.mean(res0 ** 2)) / np.radians(prior_off_deg)
                    rows, rhs = [], []
                    idx_live = np.flatnonzero(live)
                    for k in np.flatnonzero(~frozen):
                        for t in (0, 1):
                            col = 3 * k + 1 + t
                            row = np.zeros(len(idx_live))
                            row[np.searchsorted(idx_live, col)] = alpha
                            rows.append(row)
                            rhs.append(-alpha * scale[k] * off[k, t])
                    Ar, yr = np.vstack([Ar, np.array(rows)]), np.concatenate([yr, rhs])
                theta[live] = np.linalg.lstsq(Ar, yr, rcond=None)[0]
            new_scale = np.where(frozen, 1.0, theta[0::step])
            new_off = off
            if solve_pos:
                d = np.stack([theta[1::3], theta[2::3]], 1) / np.maximum(np.abs(new_scale), 1e-3)[:, None]
                lim = np.radians(max_step_deg)
                new_off = np.where(frozen[:, None], 0.0, off + np.clip(d, -lim, lim))
            bad = ~frozen & ((new_scale < [b[0] for b in scale_bounds]) | (new_scale > [b[1] for b in scale_bounds])
                             | (np.degrees(np.hypot(new_off[:, 0], new_off[:, 1])) > max_off_deg))
            frozen |= bad
            scale = np.where(frozen, 1.0, np.maximum(new_scale, 0.0))
            off = np.where(frozen[:, None], 0.0, new_off)
            if solve_pos or bad.any():
                dirs = [shift_direction(src, off[k]) for k, (src, _) in enumerate(sources)]
                P = [point_matrix(r, f, dirs[k], sources[k][1]) for k in range(K)]
        M = M_fixed + sum(scale[k] * P[k] for k in range(K))
        g_old = g
        g, _, it = stefcal(R, M, W, g0=g, niter=niter, tol=0)
        it_total += it
        if outer > 2 and np.linalg.norm(g - g_old) / np.linalg.norm(g) < tol:
            break
    return g, scale, off, dict(iterations=it_total, passes=outer + 1, frozen=np.flatnonzero(frozen).tolist())


def anchor_offsets(off, w):
    """Remove the weighted mean position offset of the bright sources: off [K, 2] (radians or
    degrees), w [K] >= 0 (for example the solved fluxes).

    A phase gradient across the stands shifts the whole image, and a common offset of all
    source positions does the same, so the two are degenerate in the solve. Anchoring the mean
    offset to zero ties the image frame to the catalogue positions and leaves only the
    differential (direction-dependent) refraction to the sources; the gains then carry any
    common shift. With fewer than two sources the offsets are zeroed."""
    off = np.asarray(off, float)
    w = np.nan_to_num(np.clip(np.asarray(w, float), 0, None))
    if off.shape[0] < 2 or w.sum() <= 0:
        return np.zeros_like(off)
    return off - (w[:, None] * off).sum(0) / w.sum()


def sky_fraction(g, ac, m_ac):
    """Fraction of each stand's power that is sky signal: |g_a|^2 M_aa / R_aa.

    g [n] solved gains, ac [n] measured autocorrelations R_aa, m_ac the model autocorrelation
    (sky power through the beam, the same for every stand). For a sky-noise-dominated stand it
    is about the same for all stands; a stand with excess receiver noise or a fault has a lower
    value. The matched-filter imaging weight is w_a = conj(g_a) / R_aa, which is
    (1 / g_a) x sky_fraction_a up to a constant: 1/g for healthy stands, down-weighted noisy ones
    (unlike plain 1/g, which up-weights weak stands and raises the image noise)."""
    ac = np.asarray(ac, float)
    return np.where(ac > 0, np.abs(g) ** 2 * float(m_ac) / np.where(ac > 0, ac, 1), 0.0)

