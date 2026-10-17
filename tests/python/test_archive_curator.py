"""
Tests of the archive curator (src/python/archive_curator.py) on temporary
directories. The database test starts a throwaway PostgreSQL server (skipped
where the PostgreSQL binaries are not installed).

    cd tests/python && python -m unittest -v test_archive_curator
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src", "python"))

import archive_curator as ac  # noqa: E402

DAY = 86400.0
NOW = 1792252800.0  # 2026-10-17T12:00:00Z
FITS_SIZE = 1000


def day_dir(t):
    return time.strftime("%Y%m%d", time.gmtime(t))


class Tree:
    """A temporary FITS archive (two nodes) and skymon archive"""

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="curator-")
        self.fits = os.path.join(self.dir, "accumulations")
        self.sky = os.path.join(self.dir, "skymon")
        self.state = os.path.join(self.dir, "state")

    def add_fits(self, node, age_d, png=True):
        t = NOW - age_d * DAY
        d = os.path.join(self.fits, node)
        os.makedirs(d, exist_ok=True)
        stem = os.path.join(d, f"EPIC_{t:.3f}_57.400MHz")
        with open(stem + ".fits", "wb") as f:
            f.write(b"\0" * FITS_SIZE)
        if png:
            with open(stem + "_QL.png", "wb") as f:
                f.write(b"\0" * 10)
        return stem + ".fits"

    def add_candidate(self, node, cid, age_d):
        d = os.path.join(self.sky, node, day_dir(NOW - age_d * DAY))
        os.makedirs(d, exist_ok=True)
        paths = []
        for ext, size in (("json", 20), ("npz", 500)):
            p = os.path.join(d, f"{cid}.{ext}")
            with open(p, "wb") as f:
                f.write(b"x" * size)
            paths.append(p)
        return paths

    def add_event(self, eid, age_d, review, bands):
        d = os.path.join(self.sky, "events", day_dir(NOW - age_d * DAY))
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, eid + ".json")
        ev = dict(id=eid, review=review, bands={n: dict(candidate=dict(id=c)) for n, c in bands.items()})
        with open(p, "w") as f:
            json.dump(ev, f)
        return p

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class FakeDB:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def apply(self, paths, dry_run):
        if self.fail:
            raise ConnectionError("database down")
        self.calls.append((list(paths), dry_run))
        return len(paths)


def fake_usage(tree, total, extra=0):
    """Usage of a pretend file system holding the tree plus `extra` bytes"""
    def usage(root):
        used = extra
        for d, _, fs in os.walk(tree.dir):
            used += sum(os.path.getsize(os.path.join(d, f)) for f in fs)
        return "fs0", used / total, used, total
    return usage


def low_usage(root):
    return "fs0", 0.10, 10, 100


class TestCurator(unittest.TestCase):
    def setUp(self):
        self.t = Tree()
        self.lines = []

    def tearDown(self):
        self.t.close()

    def curator(self, **kw):
        kw.setdefault("fits_roots", [self.t.fits] if os.path.isdir(self.t.fits) else [])
        kw.setdefault("skymon_roots", [self.t.sky] if os.path.isdir(self.t.sky) else [])
        kw.setdefault("now", NOW)
        kw.setdefault("usage_fn", low_usage)
        kw.setdefault("log", self.lines.append)
        return ac.Curator(**kw)

    def test_age_rule_by_name_not_mtime(self):
        old = self.t.add_fits("epic1", 800)
        new = self.t.add_fits("epic1", 700)
        recent = self.t.add_fits("epic2", 1)
        os.utime(new, (0, 0))  # an old mtime must not matter
        os.utime(old, (NOW, NOW))  # nor a new one
        other = os.path.join(self.t.fits, "epic1", "notes.txt")
        with open(other, "w") as f:
            f.write("not an accumulation, never removed")
        db = FakeDB()
        rc = self.curator(db=db).run()
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.exists(old))
        self.assertFalse(os.path.exists(old.replace(".fits", "_QL.png")))
        for p in (new, recent, other):
            self.assertTrue(os.path.exists(p), p)
        self.assertEqual(db.calls, [([old], False)])

    def test_dry_run_changes_nothing(self):
        old = self.t.add_fits("epic1", 800)
        self.t.add_fits("epic1", 1)
        db = FakeDB()
        c = self.curator(db=db, dry_run=True, state_dir=self.t.state)
        self.assertEqual(c.run(), 0)
        self.assertTrue(os.path.exists(old))
        self.assertFalse(os.path.exists(self.t.state))  # no lock, log or journal written
        self.assertEqual(db.calls, [([old], True)])  # rows counted, not deleted
        self.assertTrue(any("would remove age fits" in l for l in self.lines))

    def test_skymon_review_protection_under_age_rule(self):
        pend = self.t.add_candidate("epic3", "c_pending", 800)
        done = self.t.add_candidate("epic3", "c_reviewed", 800)
        orphan = self.t.add_candidate("epic3", "c_orphan", 800)
        young = self.t.add_candidate("epic3", "c_young", 10)
        ev_p = self.t.add_event("ev-p", 800, "pending", {"epic3": "c_pending"})
        ev_r = self.t.add_event("ev-r", 800, "rfi", {"epic3": "c_reviewed"})
        self.t.add_event("ev-y", 10, "frb", {"epic3": "c_young"})
        self.assertEqual(self.curator().run(), 0)
        for p in pend + young + [ev_p]:
            self.assertTrue(os.path.exists(p), p)
        for p in done + orphan + [ev_r]:
            self.assertFalse(os.path.exists(p), p)

    def test_keep_review_extends_protection(self):
        c = self.t.add_candidate("epic3", "c_frb", 800)
        ev = self.t.add_event("ev-f", 800, "frb", {"epic3": "c_frb"})
        self.t.add_event("ev-new", 1, "pending", {})
        self.curator(keep_review=["pending", "frb"]).run()
        for p in c + [ev]:
            self.assertTrue(os.path.exists(p), p)

    def test_ceiling_order_and_protection(self):
        f_old = self.t.add_fits("epic1", 300, png=False)
        f_mid = self.t.add_fits("epic2", 200, png=False)
        f_new = self.t.add_fits("epic1", 100, png=False)
        f_young = self.t.add_fits("epic1", 5, png=False)  # inside the protected period
        c_old = self.t.add_candidate("epic3", "c_old", 400)  # older, but skymon goes after FITS
        c_pend = self.t.add_candidate("epic3", "c_pend", 400)
        c_orph = self.t.add_candidate("epic3", "c_orph", 400)
        self.t.add_event("ev-o", 400, "rfi", {"epic3": "c_old"})
        self.t.add_event("ev-p", 400, "pending", {"epic3": "c_pend"})
        # 10 kB file system: ~5.7 kB of files + 3 kB elsewhere = 87%; the target
        # (83%) needs about 400 bytes, so one FITS file, the oldest
        usage = fake_usage(self.t, 10000, extra=3000)
        self.assertGreaterEqual(usage("x")[1], 0.85)
        self.assertEqual(self.curator(usage_fn=usage).run(), 0)
        self.assertFalse(os.path.exists(f_old))
        for p in [f_mid, f_new, f_young] + c_old + c_pend + c_orph:
            self.assertTrue(os.path.exists(p), p)
        # far above: everything eligible goes, never the young, pending or orphan data
        lines = []
        usage = fake_usage(self.t, 10000, extra=9000)
        rc = self.curator(usage_fn=usage, log=lines.append).run()
        self.assertEqual(rc, 1)  # still above the ceiling: reported as an error
        for p in [f_mid, f_new] + c_old:
            self.assertFalse(os.path.exists(p), p)
        for p in [f_young] + c_pend + c_orph:
            self.assertTrue(os.path.exists(p), p)
        self.assertTrue(any("nothing eligible" in l for l in lines))

    def test_clock_guard(self):
        a = self.t.add_fits("epic1", 800)
        b = self.t.add_fits("epic2", 900)
        self.assertEqual(self.curator().run(), 1)
        self.assertTrue(os.path.exists(a) and os.path.exists(b))

    def test_lock_and_stale_lock(self):
        old = self.t.add_fits("epic1", 800)
        self.t.add_fits("epic1", 1)
        os.makedirs(self.t.state)
        lock = os.path.join(self.t.state, "curator.lock")
        with open(lock, "w") as f:
            f.write(f"epic4 123 {time.time():.0f}\n")
        self.assertEqual(self.curator(state_dir=self.t.state).run(), 0)
        self.assertTrue(os.path.exists(old))  # another run holds the lock
        with open(lock, "w") as f:
            f.write(f"epic4 123 {time.time() - 13 * 3600:.0f}\n")
        self.assertEqual(self.curator(state_dir=self.t.state).run(), 0)
        self.assertFalse(os.path.exists(old))
        self.assertFalse(os.path.exists(lock))  # released
        logs = [f for f in os.listdir(self.t.state) if f.startswith("curator-")]
        self.assertEqual(len(logs), 1)
        with open(os.path.join(self.t.state, logs[0])) as f:
            self.assertIn("removed age fits", f.read())

    def test_db_outage_is_journaled_and_replayed(self):
        old = self.t.add_fits("epic1", 800)
        self.t.add_fits("epic1", 1)
        self.assertEqual(self.curator(state_dir=self.t.state, db=FakeDB(fail=True)).run(), 1)
        self.assertFalse(os.path.exists(old))
        with open(os.path.join(self.t.state, "pending-db.txt")) as f:
            self.assertEqual(f.read().split(), [old])
        db = FakeDB()
        self.assertEqual(self.curator(state_dir=self.t.state, db=db).run(), 0)
        self.assertEqual(db.calls, [([old], False)])
        self.assertFalse(os.path.exists(os.path.join(self.t.state, "pending-db.txt")))

    def test_two_runs_at_once_are_harmless(self):
        old = self.t.add_fits("epic1", 800)
        self.t.add_fits("epic1", 1)
        a, b = self.curator(), self.curator()
        a.scan_fits(self.t.fits)  # b scans the same files, then a removes them first
        items = b.scan_fits(self.t.fits)
        self.assertEqual(a.run(), 0)
        self.assertEqual(b._delete([it for it in items if it.t < NOW - 731 * DAY], "age"),
                         FITS_SIZE + 10)  # already gone: no error
        self.assertFalse(os.path.exists(old))
        self.assertEqual(b.errors, [])

    def test_max_delete(self):
        for age in (800, 801, 802):
            self.t.add_fits("epic1", age)
        self.t.add_fits("epic1", 1)
        c = self.curator(max_delete=2)
        self.assertEqual(c.run(), 1)
        self.assertEqual(len(c.removed), 2)
        self.assertEqual(len(os.listdir(os.path.join(self.t.fits, "epic1"))), 4)

    def test_operation_time_limit(self):
        t0 = time.time()
        with self.assertRaises(ac.OpTimeout):
            ac.run_limited(["sleep", "30"], 1)
        self.assertLess(time.time() - t0, 10)

    def test_hung_scan_stops_the_run(self):
        self.t.add_fits("epic1", 800)
        c = self.curator()

        def hung(root, depth):
            raise ac.OpTimeout("find did not finish in 1 s")
        c._list = hung
        self.assertEqual(c.run(), 1)
        self.assertEqual(c.removed, [])

    def test_missing_archive_is_skipped(self):
        old = self.t.add_fits("epic1", 800)
        self.t.add_fits("epic1", 1)
        c = self.curator(skymon_roots=[os.path.join(self.t.dir, "not-yet")])
        self.assertEqual(c.run(), 0)
        self.assertFalse(os.path.exists(old))
        c = self.curator(fits_roots=[os.path.join(self.t.dir, "gone")], skymon_roots=[])
        self.assertEqual(c.run(), 1)  # nothing present at all: reported

    def test_real_statfs(self):
        fsid, frac, used, total = self.curator()._usage(self.t.dir)
        self.assertTrue(0 <= frac <= 1 and total > 0 and fsid)


PG = None
try:
    from test_localdb import PGBIN, PgServer
    PG = PGBIN
except Exception:  # numpy or the helper unavailable
    PgServer = None


@unittest.skipIf(PG is None, "PostgreSQL binaries not installed")
class TestMetadataDB(unittest.TestCase):
    def test_rows_of_removed_files(self):
        pg = PgServer()
        try:
            dsn = pg.dsn()
            sql = ("CREATE SCHEMA sevilleta; CREATE TABLE sevilleta.epic_files_metadata2 "
                   "(file_name text NOT NULL, epoch_time_s integer NOT NULL); "
                   "INSERT INTO sevilleta.epic_files_metadata2 VALUES "
                   "('/a/epic1/EPIC_1.000_57.400MHz.fits', 1), "
                   "('/a/epic2//EPIC_2.000_57.400MHz.fits', 2), "
                   "('/a/epic1/EPIC_3.000_57.400MHz.fits', 3)")
            subprocess.run([os.path.join(PG, "psql"), dsn, "-q", "-c", sql], check=True, capture_output=True)
            db = ac.MetadataDB(dsn, "sevilleta")
            gone = ["/a/epic1/EPIC_1.000_57.400MHz.fits", "/a/epic2/EPIC_2.000_57.400MHz.fits"]
            self.assertEqual(db.apply(gone, dry_run=True), 2)
            self.assertEqual(db.apply(gone, dry_run=False), 2)
            out = subprocess.run([os.path.join(PG, "psql"), dsn, "-At", "-c",
                                  "SELECT epoch_time_s FROM sevilleta.epic_files_metadata2"],
                                 check=True, capture_output=True, text=True).stdout.split()
            self.assertEqual(out, ["3"])
            with self.assertRaises(ValueError):
                ac.MetadataDB(dsn, "x; DROP TABLE y")
        finally:
            pg.close()


if __name__ == "__main__":
    unittest.main()
