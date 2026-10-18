"""streamcal: CPU streaming calibration service for one epic++ instance.

    python -m streamcal.daemon --vis /dev/shm/epic_vis_gpu0 --gains /dev/shm/epic_gains_gpu0

Loop:
  1. wait for a visibility table from the epic++ calibration tap (--caltap_nseq)
  2. forward model: sky snapshot (A-team + LFSM + Sun) through the dipole beam
     at the table's mid time, refreshed every --model-refresh seconds
  3. per solved channel and polarisation: apply the LSL cable model, flag
     inputs with abnormal power, and solve the gains together with the flux and
     position of every bright compact source (A-team, Sun): scintillation,
     absorption and solar bursts change fluxes, refraction moves sources.
     The source offsets of all channels are fitted to d = d0 (f_c/f)^2 and the
     band is re-solved with positions fixed to the fit. Channels whose
     chi^2/dof shows RFI are rejected (relative to the band median)
  4. Kalman tracker per stand and polarisation (phase, delay) plus slow
     amplitude and per-channel bandpass; interpolate to all channels
  5. publish c = 1/g to the gain table (epic++ --gain_table), unless --dry-run

The gauge (overall phase per polarisation) is fixed so that the corrections
average to zero phase, which leaves the X-Y phase of the cable model as it is.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np
from astropy.time import Time

from . import forward as F
from . import rfi
from . import geometry as E
from . import shm
from . import solvers as C

SOL = 299792458.0
log = logging.getLogger("streamcal")


def cable_correction(freq_hz, pol):
    """exp(2 pi i f tau_cable) / sqrt(cable gain), as gen_phases_lwasv applies it."""
    from lsl.common.stations import lwasv

    ants = lwasv.antennas[pol::2]
    return np.array([np.exp(2j * np.pi * freq_hz * a.cable.delay(freq_hz)) / np.sqrt(a.cable.gain(freq_hz))
                     for a in ants])


def _solve_job(job):
    """One channel and polarisation: stefcal_sky (cold start: plain StEFCal first), chi^2, variances."""
    R, Md, srcs, W, r, f, g0, solve_pos, passes, fixed_pos, ac, ok, T, max_chi2 = job[:14]
    prior = job[14] if len(job) > 14 else None
    neff = job[15] if len(job) > 15 else 100.0
    rcfg = job[16] if len(job) > 16 else None  # interference projection (rfi.py), or None
    bounds = [(0.1, 1000.0) if s_[2] == "Sun" else (0.25, 4.0) for s_ in srcs]
    src2 = [(s_[0], s_[1]) for s_ in srcs]
    M0 = Md + sum(C.point_matrix(r, f, s_[0], s_[1]) for s_ in srcs)

    def sky(Rd, g_init, n_outer, pos=None):
        g_, sc_, off_, info_ = C.stefcal_sky(Rd, Md, src2, W, r, f, g0=g_init, solve_pos=solve_pos if pos is None else pos,
                                             fixed_pos=fixed_pos, n_outer=n_outer, scale_bounds=bounds,
                                             prior_off_deg=prior, prior_neff=neff, prior_nsamp=T)
        dirs_ = [C.shift_direction(s_[0], off_[j]) for j, s_ in enumerate(srcs)]
        return g_, sc_, off_, info_, Md + sum(sc_[j] * C.point_matrix(r, f, dirs_[j], srcs[j][1]) for j in range(len(srcs)))

    def pchi(P_, g_, Mfit_):
        """Median per-stand chi^2/dof of the projected residual."""
        res_ = R - np.outer(g_, g_.conj()) * Mfit_
        if P_ is not None:
            res_ = P_.apply(res_)
        d_ = np.maximum(ac, 1e-30)
        ca = (np.abs(res_) ** 2 / (np.outer(d_, d_) / T) * W).sum(1) / np.maximum(W.sum(1), 1)
        return float(np.median(ca[ok])) if ok.any() else np.inf

    keep = None  # (projector, gains, model) solved with the previous subspace, before a refresh
    Pj = None
    if rcfg is not None:
        rcfg = dict(rcfg, cold_start=g0 is None)
    if rcfg is None:
        if g0 is None:  # cold start: gains against the nominal sky before freeing the sky parameters
            g0, _, _ = C.stefcal(R, M0, W, niter=60)
        g, sc, off, info, Mfit = sky(R, g0, passes)
    else:
        # Interference projection: R = G M G^H + V V^H + Sigma. The subspace U of the station-fixed
        # interference is carried between tables (rcfg["U"]) and re-estimated from the whitened
        # residual when rcfg["refresh"]; the gains minimise ||W o P (R - G M G^H) P^H|| by a
        # majorise-minimise loop around stefcal_sky on R_fill = R_mod + P (R - R_mod) P^H.
        dh = np.sqrt(np.maximum(np.real(np.diag(R)), 1e-30))
        kmax, nmm = rcfg.get("kmax", 4), rcfg.get("mm", 4)
        Mfit = M0
        if g0 is None:  # cold start: amplitudes from the autocorrelations, subspace from the whitened data
            g = np.sqrt(np.real(np.diag(R)) / np.maximum(np.real(np.diag(M0)), 1e-30)).astype(complex)
            w, v = rfi.lowrank_offdiag(R / np.outer(dh, dh), kmax=kmax + 2)
            # Gain phases are a unitary similarity (diag(g), |g| = 1), so the whitened eigenvalues
            # of the sky part do not depend on them: a component is interference only if it is far
            # stronger than anything the sky model predicts (sky focus is no test here, since
            # unknown gain phases scramble the geometry of sky sources too).
            # sky coherence from the model alone (an upper bound: receiver noise only lowers it);
            # scaling the model to the data's cross-power would let the interference set the scale
            dm = np.sqrt(np.maximum(np.real(np.diag(M0)), 1e-30))
            Mw = M0 / np.outer(dm, dm)
            np.fill_diagonal(Mw, 0)
            wm = np.linalg.eigvalsh(Mw)[::-1]
            k = int(min(kmax, np.sum(w[:kmax] > max(4.0, 3.0 * wm[0]))))
            log.debug("rfi cold start: whitened eigenvalues %s, sky bound %.2f, k %d", np.round(w[:kmax + 2], 2), wm[0], k)
            Pj = rfi.Projector(v[:, :k], dh, np.clip(w[:k], 0, None))
            # gains against the nominal sky first (as without projection); stefcal_sky from far-off
            # gains drives the source fluxes astray
            Rp, Iest = Pj.apply(R), Pj.interference()
            for _ in range(3):
                Rmod = rfi.model_cov(g, M0, R, Iest)
                g, _, _ = C.stefcal(Rmod + Rp - Pj.apply(Rmod), M0, W, g0=g, niter=60)
            refresh = False  # the residual test needs solved gains; it runs on the next scheduled refresh
        else:
            # sanitise the starting gains (the tracker fills unsolved stands with placeholders):
            # magnitudes more than 10x from the median are reset to it, keeping the phase
            g = np.array(g0, complex)
            mag = np.abs(g)
            med = np.median(mag[ok & np.isfinite(mag) & (mag > 0)]) if np.any(ok & (mag > 0)) else 1.0
            bad = ~np.isfinite(g) | (mag > 10 * med) | (mag < med / 10)
            g[bad] = med * np.exp(1j * np.angle(np.where(np.isfinite(g[bad]), g[bad], 1.0)))
            refresh = rcfg.get("refresh", False) or rcfg.get("U") is None
            if rcfg.get("U") is not None:
                Pj = rfi.Projector(rcfg["U"], dh, rcfg.get("lam"))
                if refresh:
                    # solve with the previous subspace first, so the residual that re-estimates it
                    # comes from solved gains
                    Rp0, I0 = Pj.apply(R), Pj.interference()
                    for _ in range(nmm):
                        Rmod = rfi.model_cov(g, Mfit, R, I0)
                        g, _, _, _, Mfit = sky(Rmod + Rp0 - Pj.apply(Rmod), g, 2, pos=False)
                    keep = (Pj, g.copy(), Mfit)
        if refresh:
            # warm: the residual after the model with the current gains; only components that no
            # sky direction explains (the gains are good, so sky sources stay focused)
            Rmod = rfi.model_cov(g, Mfit, R, Pj.interference() if Pj is not None else None)
            th = rcfg.get("thresh", 0.5)  # < 0: relative to the residual's model-error floor
            # flagged stands (diag R ~ 0) carry model but no data: keep them out of the whitened residual
            okf = ok.astype(float)
            Ew = (R - Rmod) / np.outer(np.maximum(dh, 1e-30), np.maximum(dh, 1e-30)) * np.outer(okf, okf)
            kd, w, v = rfi.detect(Ew, T, th if th > 0 else 0.0, kmax=kmax,
                                  rel=-th if th < 0 else None)
            fmax = rcfg.get("focus", 0.3)
            j = [i for i in range(kd) if fmax >= 1 or rfi.sky_focus(v[:, i], r, f) < fmax]
            Pj = rfi.Projector(v[:, j], dh, np.clip(w[j], 0, None))
            log.debug("rfi refresh: %d detected, eigenvalues %s, kept %s", kd, np.round(w[:kmax + 2], 2), j)
        Rp, Iest = Pj.apply(R), Pj.interference()
        sc, off, info = np.ones(len(srcs)), np.zeros((len(srcs), 2)), dict(frozen=[])
        # a new subspace (cold start or refresh) changes the problem: give it time to converge
        for _ in range(nmm if (g0 is not None and not refresh) else 3 * nmm):
            Rmod = rfi.model_cov(g, Mfit, R, Iest)
            # source positions held (catalogue or the band law) inside the projected loop: fitting them
            # against model-filled data is unstable; fluxes stay free
            g, sc, off, info, Mfit = sky(Rmod + Rp - Pj.apply(Rmod), g, 2 if g0 is not None else passes, pos=False)
        if keep is not None and pchi(keep[0], keep[1], keep[2]) < pchi(Pj, g, Mfit):
            # do no harm: the re-estimated subspace fits worse than the previous one; keep that
            Pj, g, Mfit = keep
    d = np.maximum(ac, 1e-30)
    res = R - np.outer(g, g.conj()) * Mfit
    if Pj is not None:
        res = Pj.apply(res)
    chi_ant = (np.abs(res) ** 2 / (np.outer(d, d) / T) * W).sum(1) / np.maximum(W.sum(1), 1)
    chi = float(np.median(chi_ant[ok])) if ok.any() else float("nan")
    den = np.sum(np.abs(g[None, :] * Mfit) ** 2 * W, axis=1)
    sig2 = np.sum(np.abs(res * W) ** 2) / max(W.sum(), 1)
    # stands are judged against the channel's median: at full correlation rates the thermal
    # noise falls below the sky-model error, and chi^2/dof rises above 1 in every channel
    return dict(g=g, scale=sc, off=off, chi=chi, good=ok & (chi_ant < max_chi2 * 3 * max(chi, 1.0)),
                var=sig2 / np.maximum(den, 1e-30), frozen=info["frozen"],
                rfi_U=None if Pj is None else Pj.U.astype(np.complex64),
                rfi_lam=None if Pj is None or Pj.lam is None else Pj.lam.astype(np.float32),
                rfi_k=0 if Pj is None else Pj.k, rfi_cold=bool(rcfg is not None and (rcfg.get("cold_start") or rcfg.get("was_cold"))))


class StreamCal:
    def __init__(self, a):
        self.a = a
        self.ant = np.flatnonzero(E.eg.used_stands())
        xyz = E.eg.stand_xyz()[self.ant]
        self.bl = np.linalg.norm(xyz[:, None, :2] - xyz[None, :, :2], axis=-1)
        self.r = xyz
        self.reader = shm.VisTableReader(a.vis)
        self.writer = None
        self.chan0 = None
        self.model_t = None
        self.models = None
        self.trackers = None
        self.last_t = None
        self.history = []
        self.pool = None
        if a.workers > 1:
            import multiprocessing as mp
            self.pool = mp.get_context("fork").Pool(a.workers)

    def reset_window(self, chan0, nchan):
        a = self.a
        self.chan0, self.nchan = chan0, nchan
        self.freq_all = (chan0 + np.arange(nchan)) * E.CHAN_BW
        self.solve_idx = np.arange(a.chan_stride // 2, nchan, a.chan_stride)
        self.freq = self.freq_all[self.solve_idx]
        nu_c = self.freq_all[nchan // 2]
        self.trackers = [C.Tracker(len(self.ant), self.freq, nu_c, q_phase=np.radians(a.q_phase_deg) ** 2,
                                   q_delay=a.q_delay_ns ** 2) for _ in range(2)]
        self.cc = [np.array([cable_correction(f, p)[self.ant] for f in self.freq]) for p in range(2)]
        self.models = None
        if not a.dry_run:
            self.writer = shm.GainTableWriter(a.gains, chan0, nchan, solver="streamcal stefcal+kalman")
        log.info("spectral window chan0=%d nchan=%d: solving %d channels", chan0, nchan, len(self.solve_idx))

    def refresh_model(self, t: Time):
        """Fixed model part (diffuse sky) and the bright sources, per pol and solved channel."""
        chans = self.freq / E.CHAN_BW
        self.models = []
        for pol in ("XX", "YY"):
            snap = F.build_snapshot(t, chans, pol=pol, nside=self.a.nside, diffuse_model=getattr(self.a, "diffuse_model", "auto"))
            Md = [snap.matrix(self.ant, i, names=["diffuse"]) for i in range(len(chans))]
            srcs = [[(c.enu[0], float(c.flux[0, i]), c.name) for c in snap.comps if c.name != "diffuse"]
                    for i in range(len(chans))]
            self.models.append((Md, srcs))
        self.model_t = t

    def _prepare(self, tab, p, k):
        a = self.a
        ci = self.solve_idx[k]
        f = self.freq[k]
        R = tab.R[ci, p][np.ix_(self.ant, self.ant)].astype(np.complex128)
        cc = self.cc[p][k]
        R = cc[:, None] * R * np.conj(cc)[None, :]
        ac = np.real(np.diag(R))
        med = np.median(ac)
        ok = (ac > a.min_power * med) & (ac < a.max_power * med)
        W = ((self.bl / (SOL / f)) >= a.umin).astype(float)
        np.fill_diagonal(W, 0)
        W *= np.outer(ok, ok)
        return R, ac, ok, W

    def solve(self, tab: shm.VisTable):
        a = self.a
        n = len(self.ant)
        nk = len(self.freq)
        g = np.full((2, n, nk), np.nan, complex)
        var = np.full((2, n, nk), np.inf)
        chi = np.full((2, nk), np.nan)
        powered = np.zeros((2, n, nk), bool)
        names = [s_[2] for s_ in self.models[0][1][0]]
        K = len(names)
        scale = np.full((2, nk, K), np.nan)
        offs = np.zeros((2, nk, K, 2))
        warm = all(tr.initialised for tr in self.trackers)
        passes = a.passes_warm if warm else a.passes
        pred = [self.trackers[p].gains() if self.trackers[p].initialised else None for p in range(2)]
        prep, jobs = {}, []
        # interference projection: subspace carried per (pol, channel), re-estimated every
        # --rfi-refresh tables (the LWA-SV interferer's signature is steady for hours)
        use_rfi = getattr(a, "rfi", "auto") != "off"
        self.rfi_count = getattr(self, "rfi_count", 0) + 1
        refresh = (self.rfi_count - 1) % max(int(getattr(a, "rfi_refresh", 6)), 1) == 0
        rstate = getattr(self, "rfi_state", {})
        rcfgs = {}
        for p in range(2):
            for k in range(nk):
                if use_rfi:
                    st = rstate.get((p, k), {})
                    # a subspace from a cold start (whitened data, unsolved gains) is only a first
                    # guess: re-estimate it from the residual on the next table
                    rcfgs[(p, k)] = dict(U=st.get("U"), lam=st.get("lam"), refresh=refresh or st.get("cold", True),
                                         kmax=getattr(a, "rfi_kmax", 4), mm=getattr(a, "rfi_mm", 4),
                                         thresh=getattr(a, "rfi_thresh", 0.5), focus=getattr(a, "rfi_focus", 0.3))
                R, ac, ok, W = prep[(p, k)] = self._prepare(tab, p, k)
                powered[p, :, k] = ok
                g0 = None
                if pred[p] is not None:
                    x = pred[p][:, k]
                    g0 = np.where(np.isfinite(x) & (np.abs(x) > 1e-12), x, np.nanmedian(np.abs(x)))
                Md, srcs = self.models[p][0][k], self.models[p][1][k]
                jobs.append((R, Md, [(s_[0], s_[1], s_[2]) for s_ in srcs], W, self.r, self.freq[k], g0,
                             K > 0 and not a.no_positions, passes, None, ac, ok, tab.nsamp, a.max_chi2,
                             getattr(a, "refraction_prior_deg", 0.05) * (74e6 / self.freq[k]) ** 2 or None,
                             getattr(a, "refraction_prior_neff", 100.0), rcfgs.get((p, k))))
        # pass 1: gains, fluxes and per-channel positions
        res1 = list(self.pool.map(_solve_job, jobs)) if self.pool else [_solve_job(j) for j in jobs]
        for (p, k), r_ in zip(prep, res1):
            scale[p, k], offs[p, k] = r_["scale"], r_["off"]
        # refraction law d = d0 (f_c / f)^2 per source, weighted by the solved flux, both pols
        x = (self.freq_all[self.nchan // 2] / self.freq) ** 2
        law = np.zeros((K, 2))
        for j in range(K):
            w = np.nan_to_num(np.clip(scale[:, :, j], 0, None))
            for c in range(2):
                den = np.sum(w * x[None, :] ** 2)
                law[j, c] = np.sum(w * x[None, :] * offs[:, :, j, c]) / den if den > 0 else 0.0
        if a.no_positions:
            law[:] = 0
        elif not getattr(a, "free_frame", False):
            # anchor the frame: no common shift of all sources (it is degenerate with a phase
            # gradient in the gains); only differential refraction stays with the sources
            law = C.anchor_offsets(law, np.nanmedian(np.nan_to_num(scale), axis=(0, 1)) *
                                   np.array([np.nanmedian(np.abs(s_[1])) for s_ in self.models[0][1][0]]))
        # pool the law over tables: refraction changes over minutes, the per-table estimate is noisy
        tau = getattr(a, "refraction_tau", 600.0)
        prev = getattr(self, "law_pooled", None)
        if tau > 0 and prev is not None and prev.shape == law.shape and self.last_t is not None:
            alpha = 1.0 - np.exp(-max(tab.t_mid_unix - self.last_t, 1.0) / tau)
            law = prev + alpha * (law - prev)
        self.law_pooled = law
        # pass 2: gains and fluxes with the positions fixed to the law, warm-started from pass 1
        jobs2 = []
        for (p, k), j1, r_ in zip(prep, jobs, res1):
            r2cfg = None if j1[16] is None else dict(j1[16], U=r_["rfi_U"], lam=r_["rfi_lam"], refresh=False,
                                                       was_cold=r_.get("rfi_cold", False))
            jobs2.append(j1[:6] + (r_["g"], False, passes, law * x[k]) + j1[10:16] + (r2cfg,))
        res2 = list(self.pool.map(_solve_job, jobs2)) if self.pool else [_solve_job(j) for j in jobs2]
        self.rfi_k = np.zeros((2, nk), int)
        for (p, k), r_ in zip(prep, res2):
            scale[p, k] = r_["scale"]
            chi[p, k] = r_["chi"]
            if r_.get("rfi_U") is not None:
                rstate[(p, k)] = dict(U=r_["rfi_U"], lam=r_["rfi_lam"], cold=r_.get("rfi_cold", False))
                self.rfi_k[p, k] = r_["rfi_k"]
        self.rfi_state = rstate
        # RFI shows as channels far above the band's median chi^2/dof; the median itself exceeds 1
        # once the noise falls below the sky-model error (long integrations, full-rate GPU tap)
        self.chi_cut = a.max_chi2 * max(1.0, float(np.nanmedian(chi)) if np.isfinite(chi).any() else 1.0)
        skyfrac = np.full((2, n, nk), np.nan)
        for (p, k), r_ in zip(prep, res2):
            if not np.isfinite(chi[p, k]) or chi[p, k] > self.chi_cut:
                continue  # RFI or a model failure: leave this channel out of the update
            good = r_["good"]
            g[p, good, k] = r_["g"][good]
            var[p, good, k] = r_["var"][good]
            # sky fraction of each stand's power (for the noise-optimal weights in table())
            Md, srcs = self.models[p][0][k], self.models[p][1][k]
            m_ac = float(np.real(np.diag(Md)).mean()) + sum(float(sc_) * s_[1] for sc_, s_ in zip(r_["scale"], srcs))
            skyfrac[p, good, k] = C.sky_fraction(r_["g"], prep[(p, k)][1], m_ac)[good]
        self.update_skyfrac(skyfrac)
        sky = dict(names=names, scale=np.nanmedian(scale, axis=(0, 1)) if K else np.zeros(0),
                   offset_deg=np.degrees(law))  # offsets at the band centre
        return g, var, chi, sky, powered

    def update_skyfrac(self, skyfrac):
        """Per-stand sky fraction (median over the solved channels), smoothed over tables."""
        with np.errstate(all="ignore"):
            cur = np.nanmedian(skyfrac, axis=2)  # [pol, stand]
        prev = getattr(self, "skyfrac", None)
        if prev is None or prev.shape != cur.shape:
            self.skyfrac = cur
        else:
            self.skyfrac = np.where(np.isfinite(cur), np.where(np.isfinite(prev), 0.7 * prev + 0.3 * cur, cur), prev)

    def table(self):
        """Relative corrections: per channel and polarisation the median |c| is 1.

        The solved gains carry the absolute scale (data counts per Jy^1/2); it is kept in
        self.flux_scale and not applied, so the image bandpass and the numeric range of
        the GPU's half-precision gridding are unchanged by calibration."""
        corr = np.zeros((self.nchan, 256, 2), np.complex64)
        self.flux_scale = np.zeros((2, self.nchan))
        for p in range(2):
            gt = self.trackers[p].gains(self.freq_all)  # [ant, nchan]
            amp = np.abs(gt).mean(axis=1)
            # flag: no or abnormal power in most channels, never solved, or low gain
            flag = self.power_flag[p] | ~np.isfinite(amp)
            flag |= amp < self.a.flag_amp * np.median(amp[~flag])
            c = np.where(flag[:, None], 0, 1.0 / np.where(np.abs(gt) > 0, gt, 1))
            # noise-optimal weighting: matched filter w = conj(g) / R_aa = (1/g) x sky fraction.
            # Relative to the median stand and capped at 1, so healthy (sky-noise-dominated)
            # stands keep 1/g and stands with excess receiver noise are down-weighted instead of
            # amplified (plain 1/g raised the image noise by 8-25% on the TBF captures)
            sf = getattr(self, "skyfrac", None)
            if sf is not None and not getattr(self.a, "plain_weights", False):
                rel = sf[p] / np.nanmedian(sf[p][~flag]) if np.isfinite(sf[p][~flag]).any() else np.ones(len(self.ant))
                c = c * np.clip(np.nan_to_num(rel, nan=1.0), 0, 1)[:, None]
            norm = np.median(np.abs(c[~flag]), axis=0)  # per channel
            self.flux_scale[p] = 1.0 / np.maximum(norm, 1e-30)
            c = c / np.maximum(norm, 1e-30)[None, :]
            # gauge: corrections average to zero phase in every channel
            s_ = np.sum(c, axis=0)
            c = c * (np.conj(s_) / np.maximum(np.abs(s_), 1e-30))[None, :]
            corr[:, self.ant, p] = c.T
        return np.clip(np.abs(corr), 0, self.a.max_abs) * np.exp(1j * np.angle(corr))

    def step(self):
        tab = self.reader.poll()
        if tab is None:
            return False
        t = Time(self.a.fixed_time) if self.a.fixed_time else Time(tab.t_mid_unix, format="unix")
        if tab.chan0 != self.chan0 or tab.nchan != getattr(self, "nchan", None):
            self.reset_window(tab.chan0, tab.nchan)
        if self.models is None or abs((t - self.model_t).sec) > self.a.model_refresh:
            T0 = time.time()
            self.refresh_model(t)
            log.info("model refreshed for %s in %.1f s", t.isot, time.time() - T0)
        T0 = time.time()
        g, var, chi, sky, powered = self.solve(tab)
        self.power_flag = powered.mean(axis=2) < 0.5  # [pol, stand]
        # tracker time step from the data timestamps (not the model time, which --fixed-time freezes)
        dt = self.a.interval if self.last_t is None else max(tab.t_mid_unix - self.last_t, 1.0)
        self.last_t = tab.t_mid_unix
        for p in range(2):
            self.trackers[p].update(g[p], var[p], dt)
        nbad = int(np.sum(~np.isfinite(chi) | (chi > self.chi_cut)))
        nflag = [int(x) for x in self.power_flag.sum(axis=1)]
        msg = (f"table v{tab.version} {t.isot} nsamp {tab.nsamp} (skipped {tab.nskipped}); chi2/dof median "
               f"{np.nanmedian(chi):.2f}, RFI channels {nbad}/{chi.size}; interference components removed (median) "
               f"{float(np.median(getattr(self, 'rfi_k', np.zeros(1)))):.0f}; power-flagged X/Y {nflag[0]}/{nflag[1]}; "
               "sky " + ", ".join(f"{nm} x{sc:.2f} ({o[0]:+.2f},{o[1]:+.2f})deg" for nm, sc, o in
                                  zip(sky["names"], sky["scale"], sky["offset_deg"])) + "; "
               f"solve {time.time() - T0:.1f} s")
        if not self.a.dry_run:
            v = self.writer.publish(self.table(), t_solved_unix=tab.t_mid_unix)
            msg += f"; published gains v{v}"
        log.info(msg)
        self.history.append(dict(t=tab.t_mid_unix, chi=chi, sky=sky, x=[tr.x.copy() for tr in self.trackers],
                                 flux_scale=getattr(self, "flux_scale", None)))
        if self.a.save and len(self.history) % 30 == 0:
            np.save(self.a.save, self.history, allow_pickle=True)
        return True


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vis", default="/dev/shm/epic_vis_gpu0")
    ap.add_argument("--gains", default="/dev/shm/epic_gains_gpu0")
    ap.add_argument("--dry-run", action="store_true", help="solve and log, do not publish gains")
    ap.add_argument("--nside", type=int, default=64, help="HEALPix nside of the diffuse model")
    ap.add_argument("--diffuse-model", default="auto", choices=("auto", "lfsm", "ovrolwa", "gsm16"),
                    help="diffuse sky map; auto: OVRO-LWA at 30-80 MHz when its hybrid map is installed (sky.OVROLWA_MAPS or $STREAMCAL_OVROLWA_MAP), LFSM otherwise")
    ap.add_argument("--model-refresh", type=float, default=60.0, help="seconds between sky model updates")
    ap.add_argument("--chan-stride", type=int, default=4, help="solve every Nth channel")
    ap.add_argument("--umin", type=float, default=1.0, help="shortest baseline used (wavelengths)")
    ap.add_argument("--niter", type=int, default=40)
    ap.add_argument("--max-chi2", type=float, default=3.0,
                    help="chi^2/dof, relative to the band median when that exceeds 1, above which a channel is rejected")
    ap.add_argument("--min-power", type=float, default=0.3)
    ap.add_argument("--max-power", type=float, default=3.0)
    ap.add_argument("--flag-amp", type=float, default=0.15, help="flag stands below this fraction of median |g|")
    ap.add_argument("--max-abs", type=float, default=32.0)
    ap.add_argument("--q-phase-deg", type=float, default=2.0, help="phase random walk, deg/sqrt(s)")
    ap.add_argument("--q-delay-ns", type=float, default=0.05, help="delay random walk, ns/sqrt(s)")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--save", default="", help="np.save the solution history here")
    ap.add_argument("--passes", type=int, default=12, help="sky-then-StEFCal passes per solve (cold start)")
    ap.add_argument("--passes-warm", type=int, default=4, help="passes once the tracker predicts the gains")
    ap.add_argument("--workers", type=int, default=8, help="processes solving channels in parallel")
    ap.add_argument("--no-positions", action="store_true", help="solve source fluxes only (no refraction offsets)")
    ap.add_argument("--refraction-prior-deg", type=float, default=0.05,
                    help="1-sigma refraction prior on source offsets at 74 MHz (scales as f^-2; 0: none)")
    ap.add_argument("--refraction-prior-neff", type=float, default=100.0,
                    help="effective independent constraints for the prior: an offset d is accepted if it removes "
                         "more than (d/prior)^2/neff of the misfit (1 pins the sources)")
    ap.add_argument("--refraction-tau", type=float, default=600.0,
                    help="time constant (s) for pooling the refraction law over tables (0: no pooling)")
    ap.add_argument("--rfi", default="auto", choices=("auto", "off"),
                    help="project out interference components from each table (rfi.py) before solving")
    ap.add_argument("--rfi-kmax", type=int, default=4, help="most interference components removed")
    ap.add_argument("--rfi-mm", type=int, default=4, help="majorise-minimise iterations per solve")
    ap.add_argument("--rfi-refresh", type=int, default=6, help="re-estimate the interference subspace every N tables")
    ap.add_argument("--rfi-focus", type=float, default=0.3,
                    help="drop refreshed components whose sky-focus ratio reaches this (>= 1: keep all)")
    ap.add_argument("--rfi-thresh", type=float, default=0.5,
                    help="detection threshold on whitened residual eigenvalues; negative: that multiple of the "
                         "residual's model-error floor (median of eigenvalues 5-10)")
    ap.add_argument("--free-frame", action="store_true",
                    help="do not anchor the mean source offset to zero (default: anchored to the catalogue)")
    ap.add_argument("--plain-weights", action="store_true",
                    help="publish 1/g (default: 1/g x relative sky fraction, the noise-optimal weights)")
    ap.add_argument("--fixed-time", default="", help="evaluate the sky model at this time (tests on static data)")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    sc = StreamCal(a)
    log.info("waiting for %s", a.vis)
    while True:
        try:
            if not sc.step():
                time.sleep(0.5)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    sys.exit(main())
