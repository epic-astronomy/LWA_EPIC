"""
From search hits to stored candidates.

Hits of one search block are clustered (a burst is seen by the pixels of its
beam, at neighbouring trial DMs and times); each cluster becomes a candidate
at its brightest hit. A cluster spread over much more of the sky than a beam
is classified as RFI. A candidate that repeats one already stored (the same
burst found by two tiers or blocks) is not stored twice, and neither is one
that is an arm of a brighter burst in DM: a bright burst still exceeds the
threshold at trial DMs far from its own, smeared over the difference of the
sweeps (dm_arm_of); it is listed in the brighter candidate's dm_arms.

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
    flux: dict | None = None  # fluence and flux density from the calibration (flux.py)
    burst_fit: dict | None = None  # the burst model fitted to the dynamic spectrum (burstfit.py)
    duplicates: int = 0
    # weaker detections of this burst at other trial DMs (dm_arm_of): id, dm, snr, tier, t_bottom
    dm_arms: list = field(default_factory=list)
    members: list = field(default_factory=list)  # [hpix, snr] of the cluster's pixels
    files: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


K_DM = 4.148808e3  # s MHz^2 / (pc cm^-3)
# a weaker detection is an arm of a brighter one when its S/N is at most this
# many times what the arm would have (the estimate ignores the boxcar's
# mismatch and the tiers' resolution, both of which lower the arm)
ARM_MARGIN = 2.0
# classes merged as arms: possible bursts (RFI and followups are left alone)
ARM_CLASSES = ("candidate", "injection", "known")


def dm_arm_snr(snr: float, width_s: float, ddm: float, f_lo: float, f_hi: float) -> float:
    """
    S/N of a burst of this S/N and width searched at a DM off by ddm: the
    pulse is smeared over the difference of the sweeps, K ddm (f_lo^-2 -
    f_hi^-2), and its S/N falls as the square root of its widening.
    """
    smear = abs(ddm) * K_DM * (f_lo ** -2 - f_hi ** -2)
    return snr * math.sqrt(width_s / (width_s + smear))


def dm_arm_of(strong, weak, radius_deg: float) -> bool:
    """
    Is the weaker candidate an arm in DM of the stronger one: at its position
    (within a beam), arriving within the stronger burst's sweep smeared to the
    weaker one's DM, with no more S/N than such an arm has (ARM_MARGIN)?
    A second burst at the same position, overlapping in time and fainter than
    the arm of the first, would be taken for its arm; it stays in the data of
    the first.
    """
    if weak.snr >= strong.snr or strong.f_lo_mhz != weak.f_lo_mhz:
        return False
    r = elevation_radius(radius_deg, min(strong.el, weak.el))
    if separation_deg(strong.ra, strong.dec, weak.ra, weak.dec) > r:
        return False
    ddm = abs(strong.dm - weak.dm)
    smear = ddm * K_DM * (strong.f_lo_mhz ** -2 - strong.f_hi_mhz ** -2)
    tol = smear + 2 * (strong.width_s + weak.width_s) + 2 * max(strong.tier_dt, weak.tier_dt) + 1.0
    if abs(strong.t_bottom - weak.t_bottom) > tol:
        return False
    w = max(strong.width_s, strong.tier_dt)
    return weak.snr <= ARM_MARGIN * dm_arm_snr(strong.snr, w, ddm, strong.f_lo_mhz, strong.f_hi_mhz)


def _arm_record(c) -> dict:
    return dict(id=c.id, dm=round(float(c.dm), 3), snr=round(float(c.snr), 2), tier=int(c.tier),
                t_bottom=round(float(c.t_bottom), 3))


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
    n = len(hits)
    order = np.argsort([-h.snr for h in hits])
    vec = unit_vectors(ra, dec)
    cos_r = np.cos(np.radians(np.broadcast_to(np.asarray(radius_deg, float), (n,))))
    row = np.array([h.row for h in hits], np.int64)
    t_end = np.array([h.t_end for h in hits], np.int64)
    width = np.array([h.width for h in hits], np.int64)
    # the heads so far, in the order they were found (a hit joins the first
    # that matches); arrays grow by doubling so each test is one numpy step
    cap = 64
    h_row, h_t, h_w = (np.empty(cap, np.int64) for _ in range(3))
    h_vec, h_cos = np.empty((cap, 3)), np.empty(cap)
    nh = 0
    groups = []
    for i in order:
        if nh:
            dr = np.abs(row[i] - h_row[:nh])
            ok = (dr <= d_rows) & (np.abs(t_end[i] - h_t[:nh]) <= dr + width[i] + h_w[:nh] + 2)
            if ok.any():
                cand = np.flatnonzero(ok)
                near = h_vec[cand] @ vec[i] >= h_cos[cand]
                if near.any():
                    groups[int(cand[np.argmax(near)])].append(int(i))
                    continue
        if nh == cap:
            cap *= 2
            h_row, h_t, h_w, h_cos = (np.resize(a, cap) for a in (h_row, h_t, h_w, h_cos))
            h_vec = np.resize(h_vec, (cap, 3))
        h_row[nh], h_t[nh], h_w[nh] = row[i], t_end[i], width[i]
        h_vec[nh], h_cos[nh] = vec[i], cos_r[i]
        nh += 1
        groups.append([int(i)])
    return groups


class FixedDirections:
    """
    Detections remembered for hours by their direction above the horizon. A
    signal that keeps appearing at one azimuth and elevation while the sky
    moves on is tied to the ground or the instrument (interference from a
    fixed direction, correlated noise or crosstalk between inputs, an imaging
    artefact), never to the sky; an astrophysical source that repeats (a
    pulsar, a repeating FRB) stays at one RA and Dec instead.
    """

    def __init__(self, hours: float = 6.0):
        self.keep_s = hours * 3600.0
        self.t = np.empty(0)
        self.hor = np.empty((0, 3))  # unit vectors in (east, north, up)
        self.sky = np.empty((0, 3))

    @staticmethod
    def horizon_vector(az_deg, el_deg):
        az, el = np.radians(az_deg), np.radians(el_deg)
        return np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], -1)

    def repeats(self, t, az, el, ra, dec, radius_deg):
        """Earlier detections within radius_deg of (az, el) whose sky positions are more than two radii away"""
        if not len(self.t):
            return 0
        same_dir = self.hor @ self.horizon_vector(az, el) >= np.cos(np.radians(radius_deg))
        moved = self.sky @ unit_vectors(ra, dec) < np.cos(np.radians(2 * radius_deg))
        return int(np.count_nonzero(same_dir & moved & (self.t < t)))

    def add(self, t, az, el, ra, dec):
        keep = self.t > t - self.keep_s
        self.t = np.append(self.t[keep], t)
        self.hor = np.vstack([self.hor[keep], self.horizon_vector(az, el)])
        self.sky = np.vstack([self.sky[keep], unit_vectors(ra, dec)])


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
                           known=0, dm_arms=0)

    def _dir(self, unix: float) -> str:
        d = os.path.join(self.out_dir, "candidates",
                         datetime.fromtimestamp(unix, timezone.utc).strftime("%Y%m%d"))
        os.makedirs(d, exist_ok=True)
        return d

    def find_duplicate(self, c: Candidate, radius_deg: float):
        """
        A stored candidate that c repeats: the same burst found again by
        another tier or block peaks within a beam of the first (radius_deg is
        the beam, widened towards the horizon), at a similar DM and arrival.
        A wider radius would merge distinct bursts, and the weaker would be
        lost.
        """
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

    def find_arm(self, c: Candidate, radius_deg: float, of: bool):
        """
        of=True: a stored candidate of which c is an arm in DM (dm_arm_of);
        of=False: a stored candidate that is an arm of c.
        """
        with self.lock:
            for o in reversed(self.recent):
                if o.cls not in ARM_CLASSES:
                    continue
                if (dm_arm_of(o, c, radius_deg) if of else dm_arm_of(c, o, radius_deg)):
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
            c.dm_arms = dup.dm_arms + c.dm_arms
            if dup.injection_id and not c.injection_id:
                c.injection_id, c.cls = dup.injection_id, dup.cls
            with self.lock:
                self.recent.remove(dup)
            self.q.put(("supersede", dup, c.id))
        if dedupe and c.cls in ARM_CLASSES:
            strong = self.find_arm(c, radius_deg, of=True)
            if strong is not None:  # c is an arm of a brighter stored burst: listed on it
                strong.dm_arms.append(_arm_record(c))
                self.counts["dm_arms"] += 1
                self.q.put(("json", strong, None))
                return None
            # stored candidates that are arms of c: c takes their place, their data kept on disk
            while (arm := self.find_arm(c, radius_deg, of=False)) is not None:
                c.dm_arms = c.dm_arms + arm.dm_arms + [_arm_record(arm)]
                arm.dm_arms = []
                if arm.injection_id and not c.injection_id:
                    c.injection_id, c.cls = arm.injection_id, arm.cls
                with self.lock:
                    self.recent.remove(arm)
                self.q.put(("supersede", arm, c.id))
                self.counts["dm_arms"] += 1
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
                if kind in ("new", "npz") and extra is not None:
                    # "npz": the data again, extended past the burst (Monitor._extend_tails)
                    p = self.path(c, "npz")
                    tmp = p + ".tmp"
                    # uncompressed: float32 noise barely compresses, and zlib took a
                    # tenth of the monitor's time on live data
                    with open(tmp, "wb") as fh:
                        np.savez(fh, **extra)
                    os.replace(tmp, p)
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
