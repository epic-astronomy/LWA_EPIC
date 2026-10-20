"""
Retention for the archives on the shared file system.

Two kinds of archive are curated:

- ``fits``: the accumulated images of the imager (``--out_dir`` of epic++),
  ``EPIC_<unix time>_<cfreq>MHz.fits`` with its ``_QL.png`` quick look, one
  directory per node, in daily subdirectories ``YYYYMMDD/`` (files written
  before the subdirectories were introduced lie directly in the node's
  directory; both layouts are handled). Each FITS file is registered in the table
  ``<schema>.epic_files_metadata2`` by the disk saver.
- ``skymon``: the archive of the all-sky monitor (the ``--archive`` of
  ``skymon archive``): ``<node>/YYYYMMDD/<candidate>.{json,npz}`` and
  ``events/YYYYMMDD/<event>.json``.

Each run applies two rules.

1. Age. Data older than ``--retention-days`` (1827, five years) are removed. The
   time of a file is read from its name (the Unix time of a FITS file) or from
   its day directory (skymon, taken as the end of that day). A file whose time
   cannot be read is never removed. The modification time is not used.
2. Ceiling. While the file system holding an archive is at or above
   ``--ceiling`` (0.85) of its capacity, data are removed until it is below
   ``--target`` (ceiling - 0.02): FITS accumulations first, then skymon data,
   oldest first in each. Data younger than ``--protect-days`` (30) are never
   removed by this rule.

Never removed by either rule (kept for good):

- a skymon event of a kind listed in ``--keep-kinds`` (default ``sky``: every
  FRB event, whatever its review), and the data of its candidates;
- a skymon event whose review is listed in ``--keep-review`` (default
  ``pending``: awaiting a person's verdict), and the data of its candidates;
- a giant pulse (a candidate whose record has ``known_source.giant_pulse``, the
  Crab's), which belongs to no event (``--no-keep-giant-pulses`` to curate them).
 Under the ceiling rule, a skymon candidate whose event
is not in the archive is also kept, since its review state is unknown.

Safety:
- Every file system operation (listing, reading, removing, statfs, appending to
  the log) runs in a child process with a time limit, since the NFS mounts can
  hang. A timed-out operation ends the run; nothing is assumed removed.
- ``--dry-run`` removes nothing, writes nothing and changes no database row; it
  reports what would be removed.
- Runs on several nodes at once are safe: removing a file that is already gone
  is not an error, and with ``--state-dir`` a lock file there (created
  exclusively, broken after ``--lock-stale-hours``) lets one run proceed.
- If the age rule would remove every item of every archive (for example after
  a wrong system clock), the age rule is not applied and the run fails.
- At most ``--max-delete`` items are removed per run.

Day directories emptied by a run are removed. Removed FITS files are deleted from ``epic_files_metadata2`` (the table lists
the files on disk; no reader keeps rows for absent files). When the database
cannot be reached, the paths are kept in ``<state-dir>/pending-db.txt`` and
deleted on a later run. Every removal is logged on stdout and, with
``--state-dir``, appended to ``<state-dir>/curator-YYYYMM.log``.

    python archive_curator.py --fits <imager out_dir parent> --skymon <skymon archive> \\
        --state-dir <state dir> [--dry-run]
"""
from __future__ import annotations

import argparse
import calendar
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field

DAY = 86400.0
FITS_RE = re.compile(r"^EPIC_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)MHz(\.fits|_QL\.png)$")
DAY_RE = re.compile(r"^\d{8}$")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class OpTimeout(RuntimeError):
    """A file system operation did not finish within its time limit"""


def run_limited(argv, timeout_s, data: bytes | None = None):
    """
    Run argv in a child process with a time limit; return (returncode, stdout,
    stderr). On expiry the child is killed and OpTimeout is raised. A child that
    cannot be reaped (stuck in uninterruptible I/O on a hung mount) is left
    behind rather than waited for.
    """
    t = max(1, int(math.ceil(timeout_s)))
    p = subprocess.Popen(["timeout", "-k", "5", str(t)] + list(argv),
                         stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = p.communicate(data, timeout=t + 10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise OpTimeout(f"{argv[0]} did not finish in {t} s")
    if p.returncode in (124, 137):
        raise OpTimeout(f"{argv[0]} did not finish in {t} s")
    return p.returncode, out, err


@dataclass
class Item:
    """A unit of removal: the files of one FITS accumulation, skymon candidate or event"""
    kind: str  # fits | candidate | event
    root: str
    key: str
    t: float  # time of the data (Unix s)
    files: list = field(default_factory=list)  # [(path, size)]
    protected: bool = False  # never removed
    unknown: bool = False  # review state unknown: kept by the ceiling rule
    db_name: str | None = None  # path registered in epic_files_metadata2

    @property
    def size(self):
        return sum(s for _, s in self.files)

    @property
    def priority(self):
        return 0 if self.kind == "fits" else 1


class MetadataDB:
    """Rows of epic_files_metadata2 for removed FITS files"""

    def __init__(self, dsn: str = "", schema: str = "sevilleta", timeout_s: float = 120.0):
        if not IDENT_RE.match(schema):
            raise ValueError(f"bad schema name {schema!r}")
        self.dsn, self.schema, self.timeout_s = dsn, schema, timeout_s

    def _where(self):
        # the disk saver joins --out_dir and the name with "/", so a directory
        # given with a trailing slash leaves "//" in the stored name
        return "regexp_replace(file_name, '/+', '/', 'g') = ANY(%s)"

    def apply(self, paths, dry_run: bool) -> int:
        import psycopg2
        names = [re.sub("/+", "/", p) for p in paths]
        conn = psycopg2.connect(self.dsn, connect_timeout=10,
                                options=f"-c statement_timeout={int(self.timeout_s * 1000)}")
        n = 0
        try:
            with conn, conn.cursor() as cur:
                for i in range(0, len(names), 1000):
                    chunk = names[i:i + 1000]
                    if dry_run:
                        cur.execute(f"SELECT count(*) FROM {self.schema}.epic_files_metadata2 "
                                    f"WHERE {self._where()}", (chunk,))
                        n += cur.fetchone()[0]
                    else:
                        cur.execute(f"DELETE FROM {self.schema}.epic_files_metadata2 "
                                    f"WHERE {self._where()}", (chunk,))
                        n += cur.rowcount
        finally:
            conn.close()
        return n


class Curator:
    def __init__(self, fits_roots=(), skymon_roots=(), retention_days=1827.0, protect_days=30.0,
                 ceiling=0.85, target=None, keep_review=("pending",), dry_run=False,
                 keep_kinds=("sky",), keep_giant_pulses=True,
                 state_dir=None, db=None, op_timeout_s=60.0, scan_timeout_s=1800.0,
                 max_delete=100000, lock_stale_hours=12.0, now=None, usage_fn=None,
                 log=None):
        self.fits_roots = [os.path.abspath(r) for r in fits_roots]
        self.skymon_roots = [os.path.abspath(r) for r in skymon_roots]
        self.retention_s = retention_days * DAY
        self.protect_s = protect_days * DAY
        self.ceiling = ceiling
        self.target = ceiling - 0.02 if target is None else target
        self.keep_review = set(keep_review)
        self.keep_kinds = set(keep_kinds)
        self.keep_giant_pulses = keep_giant_pulses
        self.dry_run = dry_run
        self.state_dir = state_dir
        self.db = db
        self.op_timeout_s, self.scan_timeout_s = op_timeout_s, scan_timeout_s
        self.max_delete = max_delete
        self.lock_stale_s = lock_stale_hours * 3600
        self.now = time.time() if now is None else now
        self.usage_fn = usage_fn or self._usage
        self._log = log or (lambda line: print(line, flush=True))
        self.removed: list[Item] = []
        self.errors: list[str] = []
        self.summary: dict = {}
        self._lines: list[str] = []

    # ------------------------------------------------------------ helpers
    def log(self, msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} " \
               f"{'[dry run] ' if self.dry_run else ''}{msg}"
        self._log(line)
        self._lines.append(line)

    def error(self, msg):
        self.errors.append(msg)
        self.log("ERROR " + msg)

    def _usage(self, root):
        """(file system id, used fraction, bytes used, bytes total) as df reports them"""
        rc, out, err = run_limited(["stat", "-f", "-c", "%i %b %f %a %S", root], self.op_timeout_s)
        if rc != 0:
            raise RuntimeError(f"statfs {root}: {err.decode().strip()}")
        fsid, b, f, a, bs = out.decode().split()
        used, avail = (int(b) - int(f)) * int(bs), int(a) * int(bs)
        return fsid, used / max(1, used + avail), used, used + avail

    def _list(self, root, depth):
        """[(path relative to root, size)] of the files down to depth"""
        rc, out, err = run_limited(["find", root, "-mindepth", "1", "-maxdepth", str(depth),
                                    "-type", "f", "-printf", r"%s\t%P\n"], self.scan_timeout_s)
        if rc != 0:
            raise RuntimeError(f"listing {root}: {err.decode().strip()[:300]}")
        res = []
        for line in out.decode(errors="replace").splitlines():
            size, _, rel = line.partition("\t")
            if rel:
                res.append((rel, int(size)))
        return res

    def _read_many(self, paths):
        """{path: bytes} for small files, read in time-limited children; missing files left out"""
        res = {}
        script = 'for f; do if [ -f "$f" ]; then printf "%s\\0" "$f"; cat "$f"; printf "\\0"; fi; done'
        for i in range(0, len(paths), 200):
            chunk = paths[i:i + 200]
            rc, out, err = run_limited(["sh", "-c", script, "_"] + chunk,
                                       self.op_timeout_s + len(chunk))
            parts = out.split(b"\0")
            for j in range(0, len(parts) - 1, 2):
                res[parts[j].decode()] = parts[j + 1]
        return res

    def _remove(self, paths):
        """Remove paths; return those still present after the attempt"""
        failed = []
        for i in range(0, len(paths), 64):
            chunk = paths[i:i + 64]
            rc, _, err = run_limited(["rm", "-f", "--"] + chunk, self.op_timeout_s)
            if rc != 0:  # find out which ones, one at a time
                for p in chunk:
                    rc1, _, err1 = run_limited(["rm", "-f", "--", p], self.op_timeout_s)
                    if rc1 != 0:
                        failed.append(p)
                        self.error(f"cannot remove {p}: {err1.decode().strip()[:200]}")
        return failed

    def _append(self, path, lines):
        if not lines:
            return
        data = ("\n".join(lines) + "\n").encode()
        run_limited(["sh", "-c", 'mkdir -p "$(dirname "$1")" && cat >> "$1"', "_", path],
                    self.op_timeout_s, data)

    # ------------------------------------------------------------ scanning
    def scan_fits(self, root):
        groups: dict[str, Item] = {}
        for rel, size in self._list(root, 3):  # <node>/[YYYYMMDD/]file
            m = FITS_RE.match(os.path.basename(rel))
            if not m:
                continue
            stem = os.path.join(os.path.dirname(rel), f"EPIC_{m.group(1)}_{m.group(2)}MHz")
            path = os.path.join(root, rel)
            it = groups.setdefault(stem, Item("fits", root, stem, float(m.group(1))))
            it.files.append((path, size))
            if m.group(3) == ".fits":
                it.db_name = path
        return list(groups.values())

    def scan_skymon(self, root):
        """Items of a skymon archive. Raises when the events cannot all be read"""
        files = self._list(root, 3)
        cands: dict[tuple, Item] = {}
        events: dict[str, Item] = {}
        for rel, size in files:
            parts = rel.split("/")
            if len(parts) != 3 or not DAY_RE.match(parts[1]):
                continue
            top, day, name = parts
            try:
                t_end = calendar.timegm(time.strptime(day, "%Y%m%d")) + DAY
            except ValueError:
                continue
            ident = name.split(".", 1)[0]
            path = os.path.join(root, rel)
            if top == "events":
                it = events.setdefault(ident, Item("event", root, ident, t_end))
            else:
                it = cands.setdefault((top, ident), Item("candidate", root, f"{top}/{ident}", t_end))
            it.t = max(it.t, t_end)
            it.files.append((path, size))
        # review state of every event, from its latest archived copy
        jsons = [p for it in events.values() for p, _ in it.files if p.endswith(".json")]
        texts = self._read_many(jsons)
        referenced: dict[tuple, set] = {}
        for eid, it in events.items():
            js = [p for p, _ in it.files if p.endswith(".json")]
            if not js:
                it.protected = True  # only a partial copy: wait for the archiver
                continue
            try:
                ev = json.loads(texts[js[0]])
            except (KeyError, ValueError) as e:
                raise RuntimeError(f"cannot read event {js[0]}: {e!r}")
            review = ev.get("review") or "pending"
            it.protected = review in self.keep_review or ev.get("kind") in self.keep_kinds
            for node, b in (ev.get("bands") or {}).items():
                ids = list(b.get("others") or [])
                if isinstance(b.get("candidate"), dict) and b["candidate"].get("id"):
                    ids.append(b["candidate"]["id"])
                if b.get("data_id"):
                    ids.append(b["data_id"])
                for cid in ids:
                    referenced.setdefault((node, cid), set()).add(it.protected)
        for k, it in cands.items():
            flags = referenced.get(k)
            if flags is None:
                it.unknown = True
            elif True in flags:
                it.protected = True
        # candidates of no event: a giant pulse is kept for good, read from its record
        if self.keep_giant_pulses:
            loose = {p: it for it in cands.values() if it.unknown
                     for p, _ in it.files if p.endswith(".json")}
            texts = self._read_many(list(loose))
            for p, it in loose.items():
                try:
                    rec = json.loads(texts[p])
                except (KeyError, ValueError):
                    continue  # unreadable: not known to be a giant pulse, the rules apply
                if (rec.get("known_source") or {}).get("giant_pulse"):
                    it.protected = True
        return list(cands.values()) + list(events.values())

    # ------------------------------------------------------------ rules
    def _lock(self):
        if not self.state_dir or self.dry_run:
            return True
        path = os.path.join(self.state_dir, "curator.lock")
        body = f"{socket.gethostname()} {os.getpid()} {time.time():.0f}\n".encode()
        script = 'mkdir -p "$(dirname "$1")" && set -C && cat > "$1"'
        for _ in range(2):
            rc, _, _ = run_limited(["sh", "-c", script, "_", path], self.op_timeout_s, body)
            if rc == 0:
                self._lock_path = path
                return True
            held = self._read_many([path]).get(path, b"").decode().split()
            age = time.time() - float(held[2]) if len(held) >= 3 else 0.0
            if age > self.lock_stale_s:
                self.log(f"breaking a stale lock held by {' '.join(held[:2])} for {age / 3600:.1f} h")
                run_limited(["rm", "-f", "--", path], self.op_timeout_s)
                continue
            self.log(f"another curator holds {path} ({' '.join(held[:2])}); nothing done")
            return False
        return False

    def _unlock(self):
        p = getattr(self, "_lock_path", None)
        if p:
            run_limited(["rm", "-f", "--", p], self.op_timeout_s)

    def _delete(self, items, rule):
        """Remove items (or report them in a dry run); return the bytes freed"""
        freed = 0
        for it in items:
            if len(self.removed) >= self.max_delete:
                if not self.summary.get("capped"):
                    self.summary["capped"] = True
                    self.error(f"--max-delete {self.max_delete} reached; the rest waits for the next run")
                break
            failed = [] if self.dry_run else self._remove([p for p, _ in it.files])
            if failed:
                continue
            self.removed.append(it)
            freed += it.size
            self.log(f"{'would remove' if self.dry_run else 'removed'} {rule} {it.kind} "
                     f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(it.t))} {it.size} "
                     + " ".join(p for p, _ in it.files))
        return freed

    def _db_sync(self):
        names = [it.db_name for it in self.removed if it.db_name]
        pending_path = os.path.join(self.state_dir, "pending-db.txt") if self.state_dir else None
        pending = []
        if pending_path and not self.dry_run:
            raw = self._read_many([pending_path]).get(pending_path, b"")
            pending = [l for l in raw.decode().splitlines() if l]
        todo = pending + names
        if not todo:
            return
        if self.db is None:
            self.log(f"no database: {len(names)} rows of epic_files_metadata2 not updated")
            return
        try:
            n = self.db.apply(todo, self.dry_run)
            self.log(f"epic_files_metadata2: {'would delete' if self.dry_run else 'deleted'} {n} rows "
                     f"for {len(todo)} files")
            self.summary["db_rows"] = n
            if pending and pending_path:
                run_limited(["rm", "-f", "--", pending_path], self.op_timeout_s)
        except Exception as e:
            self.error(f"epic_files_metadata2 not updated: {e!r}")
            if pending_path and not self.dry_run:
                self._append(pending_path, names)

    def _remove_empty_days(self):
        """rmdir the day directories (YYYYMMDD) that removals have emptied; others stay"""
        if self.dry_run:
            return
        days = sorted({os.path.dirname(p) for it in self.removed for p, _ in it.files
                       if DAY_RE.match(os.path.basename(os.path.dirname(p)))})
        for i in range(0, len(days), 64):
            chunk = days[i:i + 64]
            run_limited(["rmdir", "--ignore-fail-on-non-empty", "--"] + chunk, self.op_timeout_s)
        self.summary["day_dirs_checked"] = len(days)

    def run(self) -> int:
        if not self._lock():
            return 0
        try:
            try:
                self._run()
            except OpTimeout as e:
                self.error(f"{e}; stopping (is the mount hung?)")
            try:
                self._remove_empty_days()
            except OpTimeout as e:
                self.error(f"{e}; empty day directories left (is the mount hung?)")
        finally:
            try:
                self._db_sync()
                if self.state_dir and not self.dry_run:
                    self._append(os.path.join(self.state_dir, time.strftime(
                        "curator-%Y%m.log", time.gmtime())), self._lines)
            except OpTimeout as e:
                self.errors.append(str(e))
            self._unlock()
        self.summary.update(removed=len(self.removed), bytes=sum(it.size for it in self.removed),
                            errors=len(self.errors), dry_run=self.dry_run)
        self._log(json.dumps(dict(curator=self.summary)))
        return 1 if self.errors else 0

    def _present(self, root):
        rc, _, _ = run_limited(["test", "-d", root], self.op_timeout_s)
        if rc != 0:
            self.log(f"{root}: not present, skipped")
        return rc == 0

    def _run(self):
        items_by_root: dict[str, list] = {}
        fits_roots = [r for r in self.fits_roots if self._present(r)]
        skymon_roots = [r for r in self.skymon_roots if self._present(r)]
        if not fits_roots and not skymon_roots:
            self.error("no archive present (is the shared file system mounted?)")
            return
        for root in fits_roots:
            items_by_root[root] = self.scan_fits(root)
        for root in skymon_roots:
            try:
                items_by_root[root] = self.scan_skymon(root)
            except (RuntimeError, OSError) as e:
                self.error(f"skymon archive {root} skipped: {e}")
        cutoff = self.now - self.retention_s
        everything = [it for items in items_by_root.values() for it in items]
        age_ok = True
        if everything and all(it.t < cutoff for it in everything):
            # a clock years ahead, or a wrong --retention-days, would empty the archives
            self.error("every item of every archive is older than the retention period; "
                       "check the system clock. The age rule is not applied")
            age_ok = False
        for root, items in items_by_root.items():
            old = [it for it in items if it.t < cutoff and not it.protected]
            kept = sum(1 for it in items if it.t < cutoff and it.protected)
            newest = max((it.t for it in items), default=None)
            self.log(f"{root}: {len(items)} items, {sum(it.size for it in items) / 1e9:.2f} GB, newest "
                     f"{time.strftime('%Y-%m-%d', time.gmtime(newest)) if newest else '-'}; "
                     f"{len(old)} older than {self.retention_s / DAY:.0f} d"
                     + (f" ({kept} more kept for review)" if kept else ""))
            self.summary.setdefault("roots", {})[root] = dict(
                items=len(items), bytes=sum(it.size for it in items), expired=len(old))
            if age_ok:
                old.sort(key=lambda it: it.t)
                self._delete(old, "age")
        # ceiling, per file system
        removed = set(id(it) for it in self.removed)
        by_fs: dict[str, list] = {}
        for root in items_by_root:
            fsid, frac, used, total = self.usage_fn(root)
            by_fs.setdefault(fsid, []).append(root)
            self.summary.setdefault("usage", {})[root] = round(frac, 4)
        protect_cut = self.now - self.protect_s
        for fsid, roots in by_fs.items():
            for _ in range(3):
                _, frac, used, total = self.usage_fn(roots[0])
                if frac < self.ceiling:
                    break
                need = used - self.target * total
                cand = sorted((it for r in roots for it in items_by_root[r]
                               if id(it) not in removed and it.t < protect_cut
                               and not it.protected and not it.unknown),
                              key=lambda it: (it.priority, it.t))
                pick, s = [], 0
                for it in cand:
                    if s >= need:
                        break
                    pick.append(it)
                    s += it.size
                self.log(f"file system of {', '.join(roots)} at {frac:.1%} (ceiling {self.ceiling:.0%}): "
                         f"{need / 1e9:.1f} GB to free, {len(pick)} items eligible")
                if not pick:
                    self.error("above the ceiling and nothing eligible to remove")
                    break
                self._delete(pick, "ceiling")
                removed |= set(id(it) for it in pick)
                if self.dry_run:
                    break


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fits", action="append", default=[],
                    help="directory of FITS accumulations, or the parent of the per-node directories")
    ap.add_argument("--skymon", action="append", default=[], help="archive of the all-sky monitor")
    ap.add_argument("--retention-days", type=float, default=1827.0)
    ap.add_argument("--protect-days", type=float, default=30.0)
    ap.add_argument("--ceiling", type=float, default=0.85)
    ap.add_argument("--target", type=float, default=None, help="default: ceiling - 0.02")
    ap.add_argument("--keep-kinds", default="sky",
                    help="comma-separated skymon event kinds never removed (sky: every FRB event)")
    ap.add_argument("--keep-giant-pulses", action=argparse.BooleanOptionalAction, default=True,
                    help="never remove giant pulses (candidates with known_source.giant_pulse)")
    ap.add_argument("--keep-review", default="pending",
                    help="comma-separated skymon review states never removed (pending, frb, unclear, rfi)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--state-dir", help="lock, log and pending database updates")
    ap.add_argument("--db", default="", help="libpq connection string (default: PG* environment)")
    ap.add_argument("--schema", default="sevilleta")
    ap.add_argument("--no-db", action="store_true", help="leave epic_files_metadata2 alone")
    ap.add_argument("--op-timeout", type=float, default=60.0)
    ap.add_argument("--scan-timeout", type=float, default=1800.0)
    ap.add_argument("--max-delete", type=int, default=100000)
    ap.add_argument("--lock-stale-hours", type=float, default=12.0)
    a = ap.parse_args(argv)
    if not a.fits and not a.skymon:
        ap.error("give at least one --fits or --skymon directory")
    db = None if a.no_db else MetadataDB(a.db, a.schema)
    c = Curator(a.fits, a.skymon, a.retention_days, a.protect_days, a.ceiling, a.target,
                [s for s in a.keep_review.split(",") if s], a.dry_run,
                keep_kinds=[s for s in a.keep_kinds.split(",") if s], keep_giant_pulses=a.keep_giant_pulses,
                state_dir=a.state_dir, db=db, op_timeout_s=a.op_timeout, scan_timeout_s=a.scan_timeout,
                max_delete=a.max_delete, lock_stale_hours=a.lock_stale_hours)
    return c.run()


if __name__ == "__main__":
    sys.exit(main())
