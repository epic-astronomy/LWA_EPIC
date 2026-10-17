"""
Spectrogram accumulation of every tracked pixel, for display.

The raw (not normalised) sky value of each tracked pixel is averaged over
interval_s and kept in a ring of hours long, as float16. A pixel's column is
blanked (NaN) whenever its slot is given to another pixel, so a spectrogram
never mixes two positions on the sky.
"""
from __future__ import annotations

import threading

import numpy as np


class Accumulator:
    def __init__(self, nchan: int, capacity: int, dt_img: float, interval_s: float = 10.0,
                 hours: float = 6.0):
        self.n_img = max(1, int(round(interval_s / dt_img)))
        self.interval = self.n_img * dt_img
        self.nslots = max(2, int(round(hours * 3600 / self.interval)))
        self.sum = np.zeros((nchan, capacity), np.float32)
        self.count = 0
        self.t_start = None
        self.ring = np.full((self.nslots, nchan, capacity), np.nan, np.float16)
        self.times = np.full(self.nslots, np.nan)
        self.head = 0  # next ring row
        self.filled = 0
        self.lock = threading.Lock()

    def begin(self, unix: float):
        if self.t_start is None:
            self.t_start = unix

    def end_image(self, unix_next: float):
        """Call after the image's raw values were added to self.sum"""
        self.count += 1
        if self.count >= self.n_img:
            with self.lock:
                self.ring[self.head] = (self.sum / self.count).astype(np.float16)
                self.times[self.head] = self.t_start
                self.head = (self.head + 1) % self.nslots
                self.filled = min(self.filled + 1, self.nslots)
            self.sum[:] = 0
            self.count = 0
            self.t_start = unix_next

    def reset_slots(self, slots):
        slots = np.asarray(slots, np.int64)
        if len(slots):
            with self.lock:
                self.ring[:, :, slots] = np.nan
            self.sum[:, slots] = 0

    def pixel(self, slot: int, since: float | None = None):
        """(times [T], data [T, C] float32) in time order for one slot"""
        with self.lock:
            idx = (self.head - self.filled + np.arange(self.filled)) % self.nslots
            t = self.times[idx]
            d = self.ring[idx, :, slot].astype(np.float32)
        if since is not None:
            k = t >= since
            t, d = t[k], d[k]
        return t, d

    def memory_bytes(self):
        return self.ring.nbytes + self.sum.nbytes
