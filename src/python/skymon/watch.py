"""
Targeted measurements requested by the combiner (combiner.py).

When one band detects a burst, the dispersion law predicts when it reaches
every other band: t_bottom = t_inf + K DM / f_lo^2, hours later at the lowest
frequencies. The combiner then asks the monitor of each band that can see the
position to watch for it. A watch names a position, a DM with its error and
the predicted arrival at the bottom of this band, with a time tolerance.

When the tier that covers the DM searches the predicted time, the monitor
measures the S/N at the position, whatever its value, over the delay rows
and times allowed by the tolerances. The position's pixel and its neighbours
are measured, and the best value is kept. The result is stored with the
pixel's data, as a candidate of class "followup", for the combiner to read.
A watch that cannot be measured is marked expired, with the reason.
"""
from __future__ import annotations

import threading
import time
import uuid

import numpy as np

from . import kernels
from .dedisp import K_DM

FIELDS = ("ra", "dec", "dm", "t_inf")


class Watches:
    def __init__(self, keep_s: float = 7 * 86400):
        self.lock = threading.Lock()
        self.items: dict[str, dict] = {}
        self.keep_s = keep_s

    def add(self, req: dict) -> dict:
        """Register a watch from a request: ra, dec, dm, t_inf, and optional dm_err, t_tol, id, event_id"""
        for k in FIELDS:
            if k not in req:
                raise ValueError(f"watch needs {k}")
        w = dict(id=str(req.get("id") or "w-" + uuid.uuid4().hex[:10]),
                 event_id=req.get("event_id"), ra=float(req["ra"]) % 360.0,
                 dec=float(req["dec"]), dm=float(req["dm"]), t_inf=float(req["t_inf"]),
                 dm_err=float(req.get("dm_err", 0.0)), t_tol=float(req.get("t_tol", 0.0)),
                 state="pending", created=time.time(), best=None, reason=None)
        if not (-90 <= w["dec"] <= 90) or w["dm"] < 0:
            raise ValueError("bad position or DM")
        with self.lock:
            if w["id"] in self.items:  # a repeated request is harmless
                return self.items[w["id"]]
            self.items[w["id"]] = w
            cutoff = time.time() - self.keep_s
            for k in [k for k, v in self.items.items() if v["created"] < cutoff]:
                del self.items[k]
        return w

    def pending(self):
        with self.lock:
            return [w for w in self.items.values() if w["state"] == "pending"]

    def list_live(self):
        """The watch dicts themselves (not copies), for the processing thread"""
        with self.lock:
            return list(self.items.values())

    def list(self, event_id=None):
        with self.lock:
            return [dict(w) for w in self.items.values()
                    if event_id is None or w["event_id"] == event_id]


def measure(window: np.ndarray, plan, D: int, tier, sigma: np.ndarray, widths, rows,
            t_range, detrend: bool = True, zerodm: bool = False):
    """
    S/N of a few pixels' window [C, W, k] over delay rows [r0, r1] and window
    times [t0, t1] (inclusive). Returns (snr, row, t, width_index, pixel) of the
    maximum, or None if the ranges are empty.
    """
    C, W, k = window.shape
    x = window.astype(np.float64)
    if detrend == 2:  # the mean only (Searcher detrend="mean")
        x -= x.mean(axis=1, keepdims=True)
    elif detrend:
        t = np.arange(W) - (W - 1) / 2.0
        x -= x.mean(axis=1, keepdims=True)
        x -= (np.tensordot(t, x, axes=(0, 1)) / np.sum(t * t))[:, None, :] * t[None, :, None]
    if zerodm:
        x -= x.mean(axis=0, keepdims=True)
    out = kernels.fdmt(np.ascontiguousarray(x, np.float32), plan, D)  # [D + 1, W, k]
    r0, r1 = max(rows[0], 0), min(rows[1], D)
    t0, t1 = max(t_range[0], 0), min(t_range[1], W - 1)
    if r0 > r1 or t0 > t1:
        return None
    cs = np.concatenate([np.zeros((D + 1, 1, k)), np.cumsum(out, axis=1)], axis=1)
    best = None
    for wi, w in enumerate(widths):
        lo = max(t0, w - 1)
        if lo > t1:
            continue
        ts = np.arange(lo, t1 + 1)
        box = cs[r0:r1 + 1, ts + 1, :] - cs[r0:r1 + 1, ts + 1 - w, :]  # [rows, times, k]
        s = sigma[r0:r1 + 1, wi][:, None, None]
        snr = np.where(s > 0, box / np.where(s > 0, s, 1), 0.0)
        i = np.unravel_index(np.argmax(snr), snr.shape)
        v = float(snr[i])
        if best is None or v > best[0]:
            best = (v, r0 + int(i[0]), int(ts[i[1]]), wi, int(i[2]))
    return best


def predicted_bottom(w: dict, f_lo: float) -> float:
    return w["t_inf"] + K_DM * w["dm"] / f_lo ** 2
