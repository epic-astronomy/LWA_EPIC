"""streamcal: CPU streaming calibration service for one epic++ instance.

    python -m streamcal.daemon --vis /dev/shm/epic_vis_gpu0 --gains /dev/shm/epic_gains_gpu0

Loop:
  1. wait for a visibility table from the epic++ calibration tap (--caltap_nseq)
  2. forward model: sky snapshot (A-team + LFSM + Sun) through the dipole beam
     at the table's mid time, refreshed every --model-refresh seconds
  3. per solved channel and polarisation: apply the LSL cable model, flag
     inputs with abnormal power, StEFCal against the model (Sun flux free),
     reject channels whose chi^2/dof shows RFI
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


class StreamCal:
    def __init__(self, a):
        self.a = a
        self.ant = np.flatnonzero(E.eg.used_stands())
        xyz = E.eg.stand_xyz()[self.ant]
        self.bl = np.linalg.norm(xyz[:, None, :2] - xyz[None, :, :2], axis=-1)
        self.reader = shm.VisTableReader(a.vis)
        self.writer = None
        self.chan0 = None
        self.model_t = None
        self.models = None
        self.trackers = None
        self.last_t = None
        self.history = []

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
        chans = self.freq / E.CHAN_BW
        self.models = []
        for pol in ("XX", "YY"):
            snap = F.build_snapshot(t, chans, pol=pol, nside=self.a.nside)
            M = [snap.matrix(self.ant, i) for i in range(len(chans))]
            Ms = [snap.matrix(self.ant, i, names=["Sun"]) for i in range(len(chans))]
            self.models.append((M, Ms))
        self.model_t = t

    def solve(self, tab: shm.VisTable):
        a = self.a
        n = len(self.ant)
        g = np.full((2, n, len(self.freq)), np.nan, complex)
        var = np.full((2, n, len(self.freq)), np.inf)
        chi = np.full((2, len(self.freq)), np.nan)
        sun = np.full((2, len(self.freq)), np.nan)
        powered = np.zeros((2, n, len(self.freq)), bool)
        T = tab.nsamp
        for p in range(2):
            pred = self.trackers[p].gains() if self.trackers[p].initialised else None
            for k, ci in enumerate(self.solve_idx):
                f = self.freq[k]
                R = tab.R[ci, p][np.ix_(self.ant, self.ant)].astype(np.complex128)
                cc = self.cc[p][k]
                R = cc[:, None] * R * np.conj(cc)[None, :]
                ac = np.real(np.diag(R))
                med = np.median(ac)
                ok = (ac > a.min_power * med) & (ac < a.max_power * med)
                powered[p, :, k] = ok
                W = ((self.bl / (SOL / f)) >= a.umin).astype(float)
                np.fill_diagonal(W, 0)
                W *= np.outer(ok, ok)
                M, Ms = self.models[p][0][k], self.models[p][1][k]
                g0 = pred[:, k] if pred is not None else None
                if g0 is not None:
                    g0 = np.where(np.isfinite(g0) & (np.abs(g0) > 1e-3), g0, 1.0)
                if np.abs(Ms).max() > 0:
                    gk, s, _ = C.stefcal(R, M - Ms, W, g0=g0, niter=a.niter, free=[Ms])
                    Mfit = M - Ms + s[0] * Ms
                    sun[p, k] = s[0]
                else:
                    gk, _, _ = C.stefcal(R, M, W, g0=g0, niter=a.niter)
                    Mfit = M
                d = np.maximum(ac, 1e-30)
                r2 = np.abs(R - np.outer(gk, gk.conj()) * Mfit) ** 2 / (np.outer(d, d) / T) * W
                chi_ant = r2.sum(1) / np.maximum(W.sum(1), 1)
                chi[p, k] = np.median(chi_ant[ok])
                if not np.isfinite(chi[p, k]) or chi[p, k] > a.max_chi2:
                    continue  # RFI or a model failure: leave this channel out of the update
                den = np.sum(np.abs(gk[None, :] * Mfit) ** 2 * W, axis=1)
                sig2 = np.sum(np.abs((R - np.outer(gk, gk.conj()) * Mfit) * W) ** 2) / max(W.sum(), 1)
                good = ok & (chi_ant < a.max_chi2 * 3)
                g[p, good, k] = gk[good]
                var[p, good, k] = sig2 / np.maximum(den[good], 1e-30)
        return g, var, chi, sun, powered

    def table(self):
        corr = np.zeros((self.nchan, 256, 2), np.complex64)
        for p in range(2):
            gt = self.trackers[p].gains(self.freq_all)  # [ant, nchan]
            amp = np.abs(gt).mean(axis=1)
            # flag: no or abnormal power in most channels, never solved, or low gain
            flag = self.power_flag[p] | ~np.isfinite(amp)
            flag |= amp < self.a.flag_amp * np.median(amp[~flag])
            c = np.where(flag[:, None], 0, 1.0 / np.where(np.abs(gt) > 0, gt, 1))
            # gauge: corrections average to zero phase in every channel
            s = np.sum(c, axis=0)
            c = c * (np.conj(s) / np.maximum(np.abs(s), 1e-30))[None, :]
            corr[:, self.ant, p] = c.T
        return np.clip(np.abs(corr), 0, self.a.max_abs) * np.exp(1j * np.angle(corr))

    def step(self):
        tab = self.reader.poll()
        if tab is None:
            return False
        t = Time(tab.t_mid_unix, format="unix")
        if tab.chan0 != self.chan0 or tab.nchan != getattr(self, "nchan", None):
            self.reset_window(tab.chan0, tab.nchan)
        if self.models is None or abs((t - self.model_t).sec) > self.a.model_refresh:
            T0 = time.time()
            self.refresh_model(t)
            log.info("model refreshed for %s in %.1f s", t.isot, time.time() - T0)
        T0 = time.time()
        g, var, chi, sun, powered = self.solve(tab)
        self.power_flag = powered.mean(axis=2) < 0.5  # [pol, stand]
        dt = self.a.interval if self.last_t is None else max((t - self.last_t).sec, 1.0)
        self.last_t = t
        for p in range(2):
            self.trackers[p].update(g[p], var[p], dt)
        nbad = int(np.sum(~np.isfinite(chi) | (chi > self.a.max_chi2)))
        nflag = [int(x) for x in self.power_flag.sum(axis=1)]
        msg = (f"table v{tab.version} {t.isot} nsamp {tab.nsamp} (skipped {tab.nskipped}); chi2/dof median "
               f"{np.nanmedian(chi):.2f}, RFI channels {nbad}/{chi.size}; power-flagged X/Y {nflag[0]}/{nflag[1]}; "
               f"Sun x{np.nanmedian(sun):.2f}; "
               f"solve {time.time() - T0:.1f} s")
        if not self.a.dry_run:
            v = self.writer.publish(self.table(), t_solved_unix=tab.t_mid_unix)
            msg += f"; published gains v{v}"
        log.info(msg)
        self.history.append(dict(t=t.unix, chi=chi, sun=sun, x=[tr.x.copy() for tr in self.trackers]))
        if self.a.save and len(self.history) % 30 == 0:
            np.save(self.a.save, self.history, allow_pickle=True)
        return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vis", default="/dev/shm/epic_vis_gpu0")
    ap.add_argument("--gains", default="/dev/shm/epic_gains_gpu0")
    ap.add_argument("--dry-run", action="store_true", help="solve and log, do not publish gains")
    ap.add_argument("--nside", type=int, default=64, help="HEALPix nside of the diffuse model")
    ap.add_argument("--model-refresh", type=float, default=60.0, help="seconds between sky model updates")
    ap.add_argument("--chan-stride", type=int, default=4, help="solve every Nth channel")
    ap.add_argument("--umin", type=float, default=1.0, help="shortest baseline used (wavelengths)")
    ap.add_argument("--niter", type=int, default=40)
    ap.add_argument("--max-chi2", type=float, default=3.0, help="chi^2/dof above which a channel is rejected")
    ap.add_argument("--min-power", type=float, default=0.3)
    ap.add_argument("--max-power", type=float, default=3.0)
    ap.add_argument("--flag-amp", type=float, default=0.15, help="flag stands below this fraction of median |g|")
    ap.add_argument("--max-abs", type=float, default=32.0)
    ap.add_argument("--q-phase-deg", type=float, default=2.0, help="phase random walk, deg/sqrt(s)")
    ap.add_argument("--q-delay-ns", type=float, default=0.05, help="delay random walk, ns/sqrt(s)")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--save", default="", help="np.save the solution history here")
    a = ap.parse_args(argv)
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
