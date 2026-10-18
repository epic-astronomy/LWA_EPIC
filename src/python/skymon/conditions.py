"""
Data-quality periods ("conditions") of one monitor.

Thunderstorms, solar activity and interference episodes raise the noise or
the trigger rate of a band for minutes to hours. A candidate found in such a
period deserves a different prior from one found on a quiet sky, so every
interval_s of data (10 s on the image clock, not the wall clock, so that a
monitor catching up after a stall describes the data and not the delay) the
monitor records a sample:

  noise_ratio             median over tracked pixels and channels of the
                          running variance of the search (var_tau_s, 5 min)
                          over a slow baseline of that variance (an
                          exponential average with baseline_tau_s, 1 h). It
                          starts at 1 and rises when the whole sky gets
                          noisier than it was over the last hour
  rfi_rate, cand_rate     RFI-class and all blind-search candidates per minute,
                          over the last minute of data
  solar, solar_rate       solar bursts (groups at the Sun, and flashes seen
                          over the sky at the same time), counted apart from
                          RFI, and their rate per minute
  sky_fraction_triggered  fraction of the tracked pixels with a hit in the
                          search blocks completed during the interval
  missing_fraction        fraction of the interval's images that were missing
  search_noise            the measured search noise relative to white noise
                          through the same chain, at widths of about 1, 7 and
                          20 s (the nearest tier and boxcar; list of target_s,
                          tier, width, width_s, ratio): the realised noise
                          floor at the widths of scattered bursts
  el_cyg_a, el_cas_a      elevations (deg) of Cyg A and Cas A, whose
                          scintillation and passage through the beam raise the
                          noise at those widths
  disturbed               noise_ratio > noise_ratio_max, rfi_rate >
                          rfi_rate_max, solar_rate > rfi_rate_max (reason
                          "solar") or sky_fraction_triggered >
                          sky_fraction_max; the reasons are listed

The samples of the last `hours` stay in memory (status, HTTP) and are queued
for the node's database, which the monitor's database thread drains; the
processing thread never waits for the database. Each candidate carries the
summary of the minute before it (Conditions.around).

The cost is small: the per-image work is a few counters, and every 10 s the
baseline is updated over the variance array [chan, pixel] and the median is
taken over a subsample of at most npix pixels.
"""
from __future__ import annotations

import collections
import math
import threading

import numpy as np

BLIND_CLASSES = ("candidate", "injection", "rfi")


class Conditions:
    def __init__(self, interval_s=10.0, baseline_tau_s=3600.0, hours=6.0, noise_ratio_max=1.5,
                 rfi_rate_max=2.0, sky_fraction_max=0.01, npix=1024, rate_window_s=60.0,
                 rng=None):
        self.interval = float(interval_s)
        self.a_base = 1.0 - math.exp(-self.interval / baseline_tau_s)
        self.noise_ratio_max = noise_ratio_max
        self.rfi_rate_max = rfi_rate_max
        self.sky_fraction_max = sky_fraction_max
        self.npix = npix
        self.rate_window = rate_window_s
        self.rng = rng or np.random.default_rng(0)
        n = max(1, int(round(hours * 3600 / self.interval)))
        self.lock = threading.Lock()  # guards the series against readers in other threads
        self.series = collections.deque(maxlen=n)
        # samples not yet in the database, oldest first; drained by the
        # database thread (deque appends and pops are thread-safe). Bounded
        # like the series: an outage longer than that loses the oldest.
        self.unwritten = collections.deque(maxlen=n)
        self.base = None
        self.base_ok = None
        self.restart()

    # ------------------------------------------------------------ state
    def restart(self, shape=None):
        """A new search (setup, gap): new baseline, and the open interval is dropped"""
        if shape is not None:
            self.base = np.zeros(shape, np.float32)
            self.base_ok = np.zeros(shape[1], bool)
        self.next_t = None
        self._open()

    def _open(self):
        self.n_images = 0
        self.n_missing = 0
        self.n_rfi = 0
        self.n_cand = 0
        self.n_solar = 0
        self.hit_slots = set()
        self.n_gain_changes = 0

    def reset_slots(self, slots):
        """Pixels that set give up their baseline; a pixel rising into the slot starts anew"""
        if self.base_ok is not None and len(slots):
            self.base_ok[np.asarray(slots, np.int64)] = False

    # ------------------------------------------------------------ per image (cheap)
    def due(self, unix):
        """True when an image at unix closes the open interval"""
        if self.next_t is None:
            self.next_t = (math.floor(unix / self.interval) + 1) * self.interval
            return False
        return unix >= self.next_t

    def note_image(self, missing):
        self.n_images += 1
        self.n_missing += bool(missing)

    def note_block(self, res):
        if not res.calibrating and res.hits:
            self.hit_slots.update(h.slot for h in res.hits)

    def note_gain_change(self):
        self.n_gain_changes += 1

    def note_candidate(self, cls):
        if cls == "solar":  # a solar burst: counted apart from RFI and candidates
            self.n_solar += 1
        elif cls in BLIND_CLASSES:
            self.n_cand += 1
            self.n_rfi += cls == "rfi"

    # ------------------------------------------------------------ every interval
    def sample(self, var, age, live, warmup, tracked, extra=None):
        """
        Close the open interval, ending at next_t; returns the sample. extra:
        further fields of the sample (the monitor's search noise at fixed
        widths and the elevations of Cyg A and Cas A).
        """
        t = self.next_t
        noise_ratio = self._noise_ratio(var, age, live, warmup)
        frac_trig = len(self.hit_slots) / tracked if tracked else 0.0
        n_all = self.n_images
        rec = dict(t=t, interval_s=self.interval, images=n_all, missing=self.n_missing,
                   missing_fraction=self.n_missing / n_all if n_all else None,
                   noise_ratio=noise_ratio, rfi=self.n_rfi, candidates=self.n_cand,
                   solar=self.n_solar,
                   sky_fraction_triggered=frac_trig, tracked=int(tracked),
                   gain_changes=self.n_gain_changes)
        if extra:
            rec.update(extra)
        # rates over the last minute of data, this interval included
        with self.lock:
            recent = [s for s in self.series if s["t"] > t - self.rate_window]
        n_rfi = self.n_rfi + sum(s["rfi"] for s in recent)
        n_cand = self.n_cand + sum(s["candidates"] for s in recent)
        minutes = (len(recent) + 1) * self.interval / 60.0
        n_solar = self.n_solar + sum(s.get("solar", 0) for s in recent)
        rec.update(rfi_rate=n_rfi / minutes, cand_rate=n_cand / minutes,
                   solar_rate=n_solar / minutes)
        reasons = []
        if noise_ratio is not None and noise_ratio > self.noise_ratio_max:
            reasons.append("noise")
        if rec["rfi_rate"] > self.rfi_rate_max:
            reasons.append("rfi_rate")
        if rec["solar_rate"] > self.rfi_rate_max:
            reasons.append("solar")
        if frac_trig > self.sky_fraction_max:
            reasons.append("sky_triggered")
        rec.update(disturbed=bool(reasons), reasons=reasons)
        with self.lock:
            self.series.append(rec)
        self.unwritten.append(rec)
        self.next_t = t + self.interval
        self._open()
        return rec

    def _noise_ratio(self, var, age, live, warmup):
        """Update the slow baseline of the variance and return the median ratio"""
        if self.base is None or self.base.shape != var.shape:
            self.restart(var.shape)
        warm = (live > 0) & (age > warmup)
        new = np.nonzero(warm & ~self.base_ok)[0]
        old = np.nonzero(warm & self.base_ok)[0]
        if len(new):  # a pixel's baseline starts at its variance: ratio 1
            self.base[:, new] = var[:, new]
            self.base_ok[new] = True
        if len(old):
            b = self.base[:, old]
            b += np.float32(self.a_base) * (var[:, old] - b)
            self.base[:, old] = b
        idx = np.nonzero(warm)[0]
        if len(idx) == 0:
            return None
        if len(idx) > self.npix:
            idx = np.sort(self.rng.choice(idx, self.npix, replace=False))
        b, v = self.base[:, idx], var[:, idx]
        ok = b > 0
        if not ok.any():
            return None
        return float(np.median(v[ok] / b[ok]))

    # ------------------------------------------------------------ queries
    def samples(self, since=None):
        with self.lock:
            s = list(self.series)
        if since is not None:
            s = [r for r in s if r["t"] > since]
        return s

    def latest(self):
        with self.lock:
            return self.series[-1] if self.series else None

    @staticmethod
    def summarise(samples):
        """Worst values over a set of samples"""
        if not samples:
            return dict(n_samples=0, disturbed=None)
        nr = [s["noise_ratio"] for s in samples if s["noise_ratio"] is not None]
        mf = [s["missing_fraction"] for s in samples if s["missing_fraction"] is not None]
        return dict(n_samples=len(samples), t_from=samples[0]["t"] - samples[0]["interval_s"],
                    t_to=samples[-1]["t"], max_noise_ratio=max(nr) if nr else None,
                    rfi_rate=max(s["rfi_rate"] for s in samples),
                    cand_rate=max(s["cand_rate"] for s in samples),
                    max_sky_fraction_triggered=max(s["sky_fraction_triggered"] for s in samples),
                    max_missing_fraction=max(mf) if mf else None,
                    disturbed=any(s["disturbed"] for s in samples),
                    reasons=sorted({r for s in samples for r in s["reasons"]}))

    def around(self, t, before_s=60.0):
        """
        Summary of the samples whose intervals overlap the before_s seconds
        up to t (a candidate's arrival at the bottom of the band). The
        interval holding t itself may still be open; then the minute before
        is described by the samples already closed.
        """
        with self.lock:
            sel = [s for s in self.series if t - before_s < s["t"] < t + s["interval_s"]]
        return self.summarise(sel)

    def status(self, window_s=3600.0):
        """The latest sample and a summary of the last hour, with the current disturbed run"""
        with self.lock:
            s = list(self.series)
        if not s:
            return dict(latest=None, summary=None, thresholds=self.thresholds())
        recent = [r for r in s if r["t"] > s[-1]["t"] - window_s]
        summ = self.summarise(recent)
        summ["disturbed_fraction"] = sum(r["disturbed"] for r in recent) / len(recent)
        since = None
        for r in reversed(s):
            if not r["disturbed"]:
                break
            since = r["t"] - r["interval_s"]
        summ["disturbed_since"] = since
        dist = [r["t"] for r in s if r["disturbed"]]
        summ["last_disturbed"] = dist[-1] if dist else None
        return dict(latest=s[-1], summary=summ, thresholds=self.thresholds(),
                    unwritten=len(self.unwritten))

    def thresholds(self):
        return dict(noise_ratio=self.noise_ratio_max, rfi_rate=self.rfi_rate_max,
                    sky_fraction_triggered=self.sky_fraction_max, interval_s=self.interval)

    # ------------------------------------------------------------ the database
    def write(self, db, batch=500):
        """
        Write queued samples to the node's database (database thread only).
        A sample leaves the queue only once it is written; returns how many.
        """
        done = 0
        while self.unwritten:
            chunk = list(self.unwritten)[:batch]
            if not db.put_conditions(chunk):
                break
            # pop exactly what was written: if the queue was full, the
            # processing thread may meanwhile have pushed the oldest out
            ids = {id(r) for r in chunk}
            while self.unwritten and id(self.unwritten[0]) in ids:
                self.unwritten.popleft()
            done += len(chunk)
        return done
