"""
Tests of the local databases, the combiner working through them, and the
archiver. A throwaway PostgreSQL server is started in a temporary directory
(skipped where the PostgreSQL binaries are not installed).

    cd tests/python && python -m unittest -v test_localdb
"""
import glob
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src", "python"))

PGBIN = next((d for d in sorted(glob.glob("/usr/lib/postgresql/*/bin"), reverse=True)
              if os.path.exists(os.path.join(d, "initdb"))), None)
T0 = 1792008000.0


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class PgServer:
    """A disposable PostgreSQL cluster"""

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="skymon-pg-")
        self.data = os.path.join(self.dir, "data")
        self.port = free_port()
        subprocess.run([os.path.join(PGBIN, "initdb"), "-D", self.data, "-A", "trust", "-U",
                        "skymon"], check=True, capture_output=True)
        self.start()
        for db in ("skymon", "node_a", "node_b"):
            subprocess.run([os.path.join(PGBIN, "createdb"), "-h", self.dir, "-p",
                            str(self.port), "-U", "skymon", db], check=True, capture_output=True)

    def dsn(self, db="skymon"):
        return f"host={self.dir} port={self.port} dbname={db} user=skymon"

    def start(self):
        subprocess.run([os.path.join(PGBIN, "pg_ctl"), "-D", self.data, "-w", "-l",
                        os.path.join(self.dir, "log"), "-o",
                        f"-p {self.port} -k {self.dir} -c listen_addresses=''", "start"],
                       check=True, capture_output=True)

    def stop(self):
        subprocess.run([os.path.join(PGBIN, "pg_ctl"), "-D", self.data, "-m", "fast", "-w",
                        "stop"], capture_output=True)

    def close(self):
        self.stop()
        shutil.rmtree(self.dir, ignore_errors=True)


@unittest.skipIf(PGBIN is None, "PostgreSQL binaries not installed")
class TestLocalDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pg = PgServer()

    @classmethod
    def tearDownClass(cls):
        cls.pg.close()

    def test_spool_survives_a_database_outage(self):
        from skymon.localdb import LocalDB
        d = tempfile.mkdtemp()
        try:
            db = LocalDB(self.pg.dsn(), os.path.join(d, "spool"))
            self.assertTrue(db.put_candidate(dict(id="c1", cls="candidate", t_inf=1.0, snr=9.0,
                                                  dm=10.0), "/x/c1.npz"))
            self.pg.stop()
            self.assertFalse(db.put_candidate(dict(id="c2", cls="candidate", t_inf=2.0, snr=8.5,
                                                   dm=20.0), "/x/c2.npz"))
            self.assertEqual(db.spooled(), 1)  # kept on local disk, not lost
            self.pg.start()
            self.assertEqual(db.replay(), 1)
            rows = db.execute("SELECT cid, npz_path FROM candidates ORDER BY id", fetch=True)
            self.assertEqual([r[0] for r in rows if r[0] in ("c1", "c2")], ["c1", "c2"])
            self.assertEqual(db.spooled(), 0)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_mailbox_reaches_the_monitor_and_back(self):
        from skymon.localdb import LocalDB
        from skymon.monitor import Config, Monitor
        d = tempfile.mkdtemp()
        try:
            mon = Monitor(Config(out_dir=d, db_dsn=self.pg.dsn("node_a"), inject_interval_s=0))
            db = LocalDB(self.pg.dsn("node_a"), os.path.join(d, "spool2"))
            db.execute("INSERT INTO watches (id, event_id, request) VALUES (%s, %s, %s)",
                       ("w1", "ev1", json.dumps(dict(ra=10, dec=20, dm=50, t_inf=T0))))
            db.execute("INSERT INTO injections (id, spec) VALUES (%s, %s)",
                       ("cinj-1", json.dumps(dict(ra=10, dec=20, dm=50, t_inf=T0, snr=20))))
            mon.sync_db()
            self.assertEqual([w["id"] for w in mon.watches.list()], ["w1"])
            self.assertEqual([b.id for b in mon.injector.waiting], ["cinj-1"])
            w = mon.watches.list_live()[0]
            w.update(state="expired", reason="test")
            mon.sync_db()
            rows = db.execute("SELECT state, reason FROM watches WHERE id = 'w1'", fetch=True)
            self.assertEqual(rows[0], ("expired", "test"))
            rows = db.execute("SELECT state FROM injections WHERE id = 'cinj-1'", fetch=True)
            self.assertEqual(rows[0][0], "accepted")
            self.assertIsNotNone(db.execute("SELECT value FROM status WHERE key = 'monitor'",
                                            fetch=True))
        finally:
            shutil.rmtree(d, ignore_errors=True)


@unittest.skipIf(PGBIN is None, "PostgreSQL binaries not installed")
class TestCombinerThroughDatabases(unittest.TestCase):
    def test_two_bands_confirm_archive_and_restart(self):
        from test_combiner import zenith_ish
        from skymon.archiver import Archiver
        from skymon.combiner import Combiner, CombinerConfig
        from skymon.dedisp import K_DM
        from skymon.geometry import Band
        from skymon.monitor import Config, Monitor
        from skymon.simulate import SkySimulator, tier_dt_function

        pg = PgServer()
        out = tempfile.mkdtemp()
        try:
            start = T0
            ra, dec = zenith_ish(start + 200)
            dm = 40.0
            frb = dict(id="sim-x", ra=ra, dec=dec, dm=dm, snr=25.0, width_s=0.003,
                       t_inf=start + 200 - K_DM * dm / 77.33 ** 2)
            bands = {"node_a": 3104 - 64, "node_b": 2400 - 64}
            sims, mons = {}, {}
            for name, chan0 in bands.items():
                b = Band(chan0, 64)
                sims[name] = SkySimulator(
                    chan0=chan0, nchan=64, start_unix=start, frb_per_hour=0, rfi_per_hour=0,
                    seed=len(name), frb_list=[dict(frb)],
                    tier_dt=tier_dt_function(b.edges_mhz(), 0.083592, 64, 300.0, 64))
                mons[name] = Monitor(Config(out_dir=os.path.join(out, name), db_dsn=pg.dsn(name),
                                            nside=32, dm_max=300.0, block0=64, nthreads=4,
                                            warmup_images=32, inject_interval_s=0,
                                            accum_hours=0.2,
                                            threshold=8.0 if name == "node_a" else 1e9))
            # no HTTP at all: the combiner reaches the nodes only through their databases
            cfg = CombinerConfig(nodes={n: "http://127.0.0.1:9" for n in bands},
                                 dsns={n: pg.dsn(n) for n in bands},
                                 out_dir=os.path.join(out, "comb"), settle_s=0.0)
            comb = Combiner(cfg)
            n_img = int(330 / 0.083592)
            for i in range(n_img):
                for name in bands:
                    meta, img = sims[name].next_image()
                    mons[name].process(meta, img)
                if i % 24 == 0:
                    for m in mons.values():
                        m.store.flush()
                        m.sync_db()
                if i % 60 == 0:
                    comb.poll(now=sims["node_a"]._t(i))
            for k in range(30):
                for m in mons.values():
                    m.store.flush()
                    m.sync_db()
                comb.poll(now=sims["node_a"]._t(n_img) + 60 * k)
            (ev,) = [e for e in comb.events.values()]
            self.assertEqual(ev["status"], "confirmed", json.dumps(ev["bands"], default=str))
            self.assertEqual(ev["bands"]["node_b"]["state"], "followup")
            # every node's database holds a copy of the event
            for name in bands:
                rows = mons[name].db.execute("SELECT status, version FROM events", fetch=True)
                self.assertEqual(rows[0][0], "confirmed")
            # a combiner started elsewhere, with no local files, recovers the state
            comb2 = Combiner(CombinerConfig(nodes=cfg.nodes, dsns=cfg.dsns,
                                            out_dir=os.path.join(out, "comb2")))
            self.assertEqual(comb2.events[ev["id"]]["status"], "confirmed")
            self.assertIn(("node_a", ev["bands"]["node_a"]["candidate"]["id"]), comb2.seen)

            # the archiver of node_b copies its followup, verified, and the event
            archive = os.path.join(out, "archive")
            os.makedirs(archive)
            arc = Archiver(pg.dsn("node_b"), os.path.join(out, "node_b"), archive, node="node_b")
            self.assertGreaterEqual(arc.archive_once(), 1)
            got = glob.glob(os.path.join(archive, "node_b", "*", "*.npz"))
            self.assertEqual(len(got), 1)
            self.assertTrue(glob.glob(os.path.join(archive, "events", "*", ev["id"] + ".json")))
            rows = arc.db.execute("SELECT count(*) FROM candidates WHERE archived IS NOT NULL",
                                  fetch=True)
            self.assertGreaterEqual(rows[0][0], 1)

            # an unwritable archive: nothing is marked archived, nothing is removed
            arc_a = Archiver(pg.dsn("node_a"), os.path.join(out, "node_a"), archive, node="node_a")
            os.chmod(archive, stat.S_IRUSR | stat.S_IXUSR)
            try:
                self.assertEqual(arc_a.archive_once(), 0)
                self.assertEqual(arc_a.retain_once(), 0)
            finally:
                os.chmod(archive, stat.S_IRWXU)
            self.assertGreaterEqual(arc_a.archive_once(), 1)

            # retention: unarchived non-RFI data are kept however old; certain RFI goes
            arc_a.db.execute("UPDATE candidates SET created = now() - interval '400 days'")
            arc_a.db.execute(
                "INSERT INTO candidates (cid, cls, record, created) VALUES "
                "('rfi-old', 'rfi', %s, now() - interval '10 days'), "
                "('cand-old', 'candidate', %s, now() - interval '400 days')",
                (json.dumps(dict(id="rfi-old", rfi_certain=True)), json.dumps(dict(id="cand-old"))))
            arc_a.keep_days = 1e6  # keep archived data for this check
            arc_a.retain_once()
            left = {r[0] for r in arc_a.db.execute("SELECT cid FROM candidates", fetch=True)}
            self.assertNotIn("rfi-old", left)
            self.assertIn("cand-old", left)
        finally:
            pg.close()
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
