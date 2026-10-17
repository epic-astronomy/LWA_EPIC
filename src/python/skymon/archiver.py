"""
Archiving and local retention on one imaging node.

The shared file system is the archive: it holds what is
analysed and must be kept, and nothing in the real-time path depends on it.
On each node the archiver, a process of its own, does two things.

1. **Archive.** It copies to <archive>/<node>/YYYYMMDD/ the JSON record and
   data products of every candidate of this node that belongs to an event
   worth keeping:
   - a confirmed event;
   - an event a reviewer judged to be an FRB, or unclear;
   - a coordinated injection.

   The selection reads the event copies in the node's own database. Each copy
   is verified by size and SHA-256 before the row is marked archived. When the
   archive is not mounted or does not answer, it waits and retries; the local
   data stay.

2. **Retain.** It frees local disk only by these rules:
   - candidates of certain RFI (no data products in any case) after rfi_days;
   - candidates of events reviewed as RFI, and RFI candidates in no event,
     after review_days;
   - candidates at known sources (class "known": the Sun, Jupiter, bright
     sources, pulsars), unless archived, after review_days;
   - archived candidates after keep_days;
   - above max_disk_fraction, archived candidates are removed first, oldest
     first.

   A candidate that might be a burst and is not archived is never removed.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time

from .localdb import LocalDB


def _archive_alive(root: str, timeout_s: float = 10.0) -> bool:
    """Is the archive mounted and writable? Checked in a child process so a hung mount cannot hang us"""
    try:
        r = subprocess.run(["timeout", str(int(timeout_s)), "test", "-w", root],
                           capture_output=True, timeout=timeout_s + 5)
        return r.returncode == 0
    except Exception:
        return False


def _copy_verified(src: str, dst_dir: str, timeout_s: float = 300.0) -> str:
    """
    Copy src into dst_dir and compare the copy byte for byte, all in one child
    process with a time limit, so that a hung archive mount can only stall the
    child. The copy appears under its final name only once it matches.
    """
    dst = os.path.join(dst_dir, os.path.basename(src))
    script = 'mkdir -p "$1" && cp --preserve=timestamps "$2" "$3.part" && cmp -s "$2" "$3.part" ' \
             '&& mv -f "$3.part" "$3"'
    r = subprocess.run(["timeout", str(int(timeout_s)), "sh", "-c", script, "_", dst_dir, src, dst],
                       capture_output=True, timeout=timeout_s + 10)
    if r.returncode != 0:
        raise IOError(f"copy to {dst} failed ({r.returncode}): {r.stderr.decode()[:200]}")
    return dst


class Archiver:
    def __init__(self, dsn: str, data_dir: str, archive_root: str, node: str | None = None,
                 rfi_days: float = 3.0, review_days: float = 14.0, keep_days: float = 30.0,
                 max_disk_fraction: float = 0.8):
        self.db = LocalDB(dsn, os.path.join(data_dir, "spool-archiver"))
        self.data_dir = data_dir
        self.root = archive_root
        self.node = node or socket.gethostname()
        self.rfi_days, self.review_days, self.keep_days = rfi_days, review_days, keep_days
        self.max_disk = max_disk_fraction
        self.stats = dict(archived=0, failed=0, removed=0, last_archive_ok=None)

    # ------------------------------------------------------------ selection
    def _wanted(self):
        """Candidate ids of this node in events worth keeping (from the local event copies)"""
        rows = self.db.execute(
            "SELECT state FROM events WHERE status = 'confirmed' OR review IN ('frb', 'unclear') "
            "OR state->>'kind' = 'injection'", fetch=True) or []
        ids = set()
        for (ev,) in rows:
            b = (ev.get("bands") or {}).get(self.node) or {}
            if b.get("candidate"):
                ids.add(b["candidate"]["id"])
            ids.update(b.get("others", []))
            if b.get("data_id"):
                ids.add(b["data_id"])
        return ids

    def archive_once(self, limit: int = 200) -> int:
        if not _archive_alive(self.root):
            return 0
        wanted = self._wanted()
        if not wanted:
            return 0
        rows = self.db.execute(
            "SELECT cid, record, npz_path FROM candidates WHERE archived IS NULL AND cid = ANY(%s) "
            "ORDER BY id LIMIT %s", (list(wanted), limit), fetch=True) or []
        n = 0
        for cid, rec, npz in rows:
            day = time.strftime("%Y%m%d", time.gmtime(rec.get("t_inf") or time.time()))
            dst = os.path.join(self.root, self.node, day)
            try:
                jpath = (rec.get("files") or {}).get("json")
                for p in (jpath, npz):
                    if p and os.path.exists(p):
                        _copy_verified(p, dst)
                self.db.execute("UPDATE candidates SET archived = now() WHERE cid = %s", (cid,))
                n += 1
            except Exception as e:
                self.stats["failed"] += 1
                print(f"archiver: {cid} not archived: {e}", flush=True)
                if not _archive_alive(self.root):
                    break
        # the events themselves, so the archive is complete without the nodes: each
        # version is written to a local staging directory and copied like the data
        evs = self.db.execute(
            "SELECT id, version, state FROM events WHERE status = 'confirmed' "
            "OR review IN ('frb', 'unclear') OR state->>'kind' = 'injection'", fetch=True) or []
        done = self._archived_versions()
        for eid, ver, ev in evs:
            if done.get(eid, 0) >= ver:
                continue
            day = time.strftime("%Y%m%d", time.gmtime(ev.get("t_inf") or time.time()))
            stage = os.path.join(self.data_dir, "archive-staging", "events", day)
            os.makedirs(stage, exist_ok=True)
            p = os.path.join(stage, eid + ".json")
            with open(p, "w") as f:
                json.dump(ev, f, indent=1)
            try:
                _copy_verified(p, os.path.join(self.root, "events", day))
                done[eid] = ver
            except Exception as e:
                print(f"archiver: event {eid} not archived: {e}", flush=True)
                break
        self._archived_versions(done)
        self.stats["archived"] += n
        self.stats["last_archive_ok"] = time.time()
        return n

    def _archived_versions(self, update=None):
        """Local index of the event versions already in the archive"""
        p = os.path.join(self.data_dir, "archive-staging", "events.json")
        if update is not None:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p + ".tmp", "w") as f:
                json.dump(update, f)
            os.replace(p + ".tmp", p)
            return update
        try:
            with open(p) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    # ------------------------------------------------------------ retention
    def _remove(self, cid, rec, npz):
        for p in ((rec.get("files") or {}).get("json"), npz):
            if p and os.path.exists(p):
                os.remove(p)
        self.db.execute("DELETE FROM candidates WHERE cid = %s", (cid,))
        self.stats["removed"] += 1

    def retain_once(self) -> int:
        now = time.time()
        rfi_reviewed = set()
        for (ev,) in self.db.execute(
                "SELECT state FROM events WHERE review = 'rfi'", fetch=True) or []:
            b = (ev.get("bands") or {}).get(self.node) or {}
            for cid in [b.get("candidate", {}).get("id"), b.get("data_id")] + b.get("others", []):
                if cid:
                    rfi_reviewed.add(cid)
        in_events = set()
        for (ev,) in self.db.execute("SELECT state FROM events", fetch=True) or []:
            b = (ev.get("bands") or {}).get(self.node) or {}
            for cid in [b.get("candidate", {}).get("id"), b.get("data_id")] + b.get("others", []):
                if cid:
                    in_events.add(cid)
        rows = self.db.execute(
            "SELECT cid, cls, record, npz_path, archived, extract(epoch from created) "
            "FROM candidates ORDER BY id", fetch=True) or []
        removed = 0
        for cid, cls, rec, npz, archived, created in rows:
            age_d = (now - float(created)) / 86400
            certain_rfi = cls == "rfi" and rec.get("rfi_certain")
            if (certain_rfi and age_d > self.rfi_days) or \
                    (cid in rfi_reviewed and not archived and age_d > self.review_days) or \
                    (cls == "rfi" and cid not in in_events and age_d > self.review_days) or \
                    (cls == "known" and not archived and age_d > self.review_days) or \
                    (archived is not None and age_d > self.keep_days):
                self._remove(cid, rec, npz)
                removed += 1
        # disk above the limit: archived data first, oldest first
        while self._disk_fraction() > self.max_disk:
            r = self.db.execute(
                "SELECT cid, record, npz_path FROM candidates WHERE archived IS NOT NULL "
                "ORDER BY id LIMIT 100", fetch=True)
            if not r:
                print("archiver: local disk above its limit and nothing archived to remove",
                      flush=True)
                break
            for cid, rec, npz in r:
                self._remove(cid, rec, npz)
                removed += 1
        return removed

    def _disk_fraction(self):
        u = shutil.disk_usage(self.data_dir)
        return u.used / u.total

    def run(self, stop, every_s: float = 60.0):
        while not stop.is_set():
            try:
                if self.db.spooled():
                    self.db.replay()
                n = self.archive_once()
                r = self.retain_once()
                self.db.put_status("archiver", dict(self.stats, archive_alive=_archive_alive(
                    self.root), disk_fraction=round(self._disk_fraction(), 3)))
                if n or r:
                    print(f"archiver: archived {n}, removed {r}", flush=True)
            except Exception as e:
                print(f"archiver: {e!r}", flush=True)
            stop.wait(every_s)
