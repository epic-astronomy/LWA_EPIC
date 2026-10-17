"""
From search hits to stored candidates.

Hits of one search block are clustered (a burst is seen by the pixels of its
beam, at neighbouring trial DMs and times); each cluster becomes a candidate
at its brightest hit. A cluster spread over much more of the sky than a beam
is classified as RFI. A candidate that repeats one already stored (the same
burst found by two tiers or blocks) is not stored twice.

Every stored candidate is a JSON record plus an .npz with the data around
it, under <out_dir>/candidates/YYYYMMDD/. Only RFI that is certain (a DM near
zero, or much of the sky at once) is stored without its data: anything that
might be a burst is kept for a person to judge. A candidate at a known
source (the Sun, Jupiter, a bright source, or a pulsar at its DM; known.py)
is stored with its data as class "known".
"""
from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np


def unit_vectors(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def separation_deg(ra1, dec1, ra2, dec2):
    v1, v2 = unit_vectors(ra1, dec1), unit_vectors(ra2, dec2)
    return np.degrees(np.arccos(np.clip(np.sum(v1 * v2, -1), -1, 1)))


def utc(unix: float) -> str:
    return datetime.fromtimestamp(unix, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass
class Candidate:
    id: str
    cls: str  # candidate | known | injection | rfi | followup | context
    snr: float
    dm: float
    dm_step: float
    width_s: float
    t_inf: float
    t_bottom: float
    t_top: float
    utc_inf: str
    hpix: int
    nside: int
    ra: float
    dec: float
    el: float
    az: float
    image_x: float
    image_y: float
    tier: int
    tier_dt: float
    n_pix: int
    extent_deg: float
    f_lo_mhz: float
    f_hi_mhz: float
    host: str
    detected_unix: float
    injection_id: str | None = None
    event_id: str | None = None  # the combiner's event, for followup measurements
    rfi_reasons: list = field(default_factory=list)  # why it was classified as RFI
    rfi_certain: bool = False  # only certain RFI is stored without its data
    quality: dict = field(default_factory=dict)  # in-band measures (quality.py)
    known_source: dict | None = None  # a known source at the position (known.py)
    duplicates: int = 0
    members: list = field(default_factory=list)  # [hpix, snr] of the cluster's pixels
    files: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def elevation_radius(radius_deg, el_deg):
    """
    The synthesised beam is fixed in direction cosines, so on the sky it
    stretches by 1 / sin(el) towards the horizon; widen a radius accordingly
    (at most four times).
    """
    return radius_deg / np.clip(np.sin(np.radians(el_deg)), 0.25, 1.0)


def cluster_hits(hits, ra, dec, radius_deg, d_rows):
    """
    Greedy clustering of one block's hits, brightest first. Two hits join when
    they are within radius_deg on the sky (a scalar, or one per hit, taken at
    the brighter hit), d_rows delay rows and a time window that grows with the
    row difference (a wrong DM moves the peak).
    Returns a list of lists of hit indices, brightest hit first.
    """
    if not hits:
        return []
    order = np.argsort([-h.snr for h in hits])
    vec = unit_vectors(ra, dec)
    cos_r = np.cos(np.radians(np.broadcast_to(np.asarray(radius_deg, float), (len(hits),))))
    heads, groups = [], []
    for i in order:
        h = hits[i]
        joined = False
        for g, k in enumerate(heads):
            b = hits[k]
            if abs(h.row - b.row) > d_rows:
                continue
            tol = abs(h.row - b.row) + h.width + b.width + 2
            if abs(h.t_end - b.t_end) > tol:
                continue
            if float(vec[i] @ vec[k]) < cos_r[k]:
                continue
            groups[g].append(i)
            joined = True
            break
        if not joined:
            heads.append(i)
            groups.append([i])
    return groups


class CandidateStore:
    """Writes candidates to disk in a background thread and keeps recent ones in memory"""

    def __init__(self, out_dir: str, keep_hours: float = 24.0, db=None):
        self.out_dir = out_dir
        self.db = db  # localdb.LocalDB: every record also goes to the node's database
        self.keep_s = keep_hours * 3600
        self.recent: list[Candidate] = []
        self.lock = threading.Lock()
        self.q: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._writer, daemon=True)
        self.thread.start()
        os.makedirs(os.path.join(out_dir, "candidates"), exist_ok=True)
        self.counts = dict(stored=0, duplicates=0, rfi=0, injection=0, followup=0, context=0,
                           known=0)

    def _dir(self, unix: float) -> str:
        d = os.path.join(self.out_dir, "candidates",
                         datetime.fromtimestamp(unix, timezone.utc).strftime("%Y%m%d"))
        os.makedirs(d, exist_ok=True)
        return d

    def find_duplicate(self, c: Candidate, radius_deg: float):
        with self.lock:
            for o in reversed(self.recent):
                if o.cls == "followup":
                    continue
                if abs(o.t_bottom - c.t_bottom) > max(30.0, 4 * (o.width_s + c.width_s)) + \
                        2 * max(o.tier_dt, c.tier_dt):
                    continue
                dm_tol = max(0.15 * max(o.dm, c.dm), 3 * max(o.dm_step, c.dm_step))
                if abs(o.dm - c.dm) > dm_tol:
                    continue
                r = elevation_radius(radius_deg, min(o.el, c.el))
                if separation_deg(o.ra, o.dec, c.ra, c.dec) > r:
                    continue
                return o
        return None

    def add(self, c: Candidate, products: dict | None, radius_deg: float, dedupe: bool = True):
        """Store c unless it repeats a brighter stored candidate. Returns the stored one or None"""
        dup = self.find_duplicate(c, radius_deg) if dedupe else None
        if dup is not None:
            if dup.snr >= c.snr:
                dup.duplicates += 1
                self.counts["duplicates"] += 1
                self.q.put(("json", dup, None))
                return None
            c.duplicates = dup.duplicates + 1
            if dup.injection_id and not c.injection_id:
                c.injection_id, c.cls = dup.injection_id, dup.cls
            with self.lock:
                self.recent.remove(dup)
            self.q.put(("supersede", dup, c.id))
        with self.lock:
            self.recent.append(c)
            cutoff = time.time() - self.keep_s
            self.recent = [o for o in self.recent if o.detected_unix > cutoff]
        self.counts["stored"] += 1
        if c.cls in self.counts:
            self.counts[c.cls] += 1
        self.q.put(("new", c, products))
        return c

    def list(self, since: float = 0.0, cls: str | None = None, limit: int = 500):
        with self.lock:
            out = [o for o in self.recent if o.t_inf >= since and (cls is None or o.cls == cls)]
        return out[-limit:]

    def get(self, cid: str):
        with self.lock:
            for o in self.recent:
                if o.id == cid:
                    return o
        return None

    def path(self, c: Candidate, ext: str) -> str:
        return os.path.join(self._dir(c.t_inf), f"{c.id}.{ext}")

    def _writer(self):
        while True:
            kind, c, extra = self.q.get()
            try:
                if kind == "new" and extra is not None:
                    p = self.path(c, "npz")
                    np.savez_compressed(p, **extra)
                    c.files["npz"] = p
                if kind == "supersede":
                    c.files["superseded_by"] = extra
                p = self.path(c, "json")
                c.files["json"] = p
                tmp = p + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(c.to_dict(), f, indent=1)
                os.replace(tmp, p)
                if self.db is not None:  # spooled locally if the database is away
                    self.db.put_candidate(c.to_dict(), c.files.get("npz"))
            except Exception as e:  # never let disk trouble stop the search
                print(f"skymon: could not write candidate {c.id}: {e}", flush=True)
            finally:
                self.q.task_done()

    def flush(self):
        self.q.join()
