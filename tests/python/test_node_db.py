"""
Tests of the imager's Python database paths with the per-node databases:
the watch list (watchdog, head database, cached copy) and the daily digest
writer. A throwaway PostgreSQL server is started in a temporary directory;
the tests that need it are skipped where the PostgreSQL binaries are not
installed. Run on an imaging node, not the head node:

    cd tests/python && python -m unittest -v test_node_db
"""
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src", "python"))

import pixel_extractor  # noqa: E402
import daily_digest  # noqa: E402

PGBIN = next((d for d in sorted(glob.glob("/usr/lib/postgresql/*/bin"), reverse=True)
              if os.path.exists(os.path.join(d, "initdb"))), None)

WATCHDOG_DDL = """
CREATE TABLE public.epic_watchdog (
    id serial PRIMARY KEY,
    source text NOT NULL,
    event_time timestamp without time zone NOT NULL,
    ra_deg double precision NOT NULL,
    dec_deg double precision NOT NULL,
    event_type text NOT NULL,
    t_start timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    t_end timestamp without time zone,
    watch_mode text DEFAULT 'continuous'::text,
    patch_type integer DEFAULT 5,
    reason text NOT NULL,
    author text NOT NULL,
    watch_status text DEFAULT 'watching'::text,
    voevent xml
);
"""

DIGEST_DDL = """
CREATE SCHEMA sevilleta;
CREATE TABLE sevilleta.epic_daily_digest (
    source_name text NOT NULL,
    img_time timestamp without time zone NOT NULL,
    stokes_i real[],
    stokes_v real[],
    cfreq real NOT NULL
);
"""


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class PgServer:
    """A disposable PostgreSQL cluster on 127.0.0.1"""

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="epic-node-db-test-")
        self.port = free_port()
        subprocess.run([f"{PGBIN}/initdb", "-D", f"{self.dir}/data", "-U", "postgres",
                        "--auth=trust", "-E", "UTF8"], check=True, capture_output=True)
        with open(f"{self.dir}/data/postgresql.conf", "a") as f:
            f.write(f"port = {self.port}\nlisten_addresses = '127.0.0.1'\n"
                    f"unix_socket_directories = '{self.dir}'\ntimezone = 'UTC'\n"
                    "fsync = off\n")
        self.start()
        self.psql("CREATE DATABASE epic", db="postgres")

    def start(self):
        subprocess.run([f"{PGBIN}/pg_ctl", "-D", f"{self.dir}/data", "-l", f"{self.dir}/log",
                        "-w", "start"], check=True, capture_output=True)

    def stop(self):
        subprocess.run([f"{PGBIN}/pg_ctl", "-D", f"{self.dir}/data", "-m", "fast", "-w",
                        "stop"], capture_output=True)

    def dsn(self, db="epic"):
        return f"host=127.0.0.1 port={self.port} dbname={db} user=postgres"

    def psql(self, sql, db="epic"):
        out = subprocess.run([f"{PGBIN}/psql", "-X", "-A", "-t", "-v", "ON_ERROR_STOP=1",
                              "-h", self.dir, "-p", str(self.port), "-U", "postgres",
                              "-d", db, "-c", sql], check=True, capture_output=True, text=True)
        return out.stdout.strip()

    def close(self):
        self.stop()
        shutil.rmtree(self.dir, ignore_errors=True)


def reset_watch_state():
    pixel_extractor._watch.update(df=None, json=None, source=None, time=0.0,
                                  grpc_retry_at=0.0, log_at={})


def watch_df(names):
    return pd.DataFrame({"id": list(range(len(names))), "source_name": names,
                         "ra": [10.0 * i for i in range(len(names))],
                         "dec": [20.0] * len(names), "kernel_dim": [5] * len(names)})


class WatchListNoDatabase(unittest.TestCase):
    def setUp(self):
        reset_watch_state()
        self.tmp = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {
            "EPIC_WATCHLIST_CACHE": os.path.join(self.tmp, "watchlist.json"),
            "EPIC_WATCH_DSN": ""})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)
        reset_watch_state()

    def test_cached_copy_survives_a_restart(self):
        with mock.patch.object(pixel_extractor, "_fetch_watchdog",
                               return_value=watch_df(["sun", "Cyg A"])):
            df = pixel_extractor.get_watch_list("x:1")
        self.assertEqual(list(df.source_name), ["sun", "Cyg A"])
        self.assertTrue(os.path.exists(os.environ["EPIC_WATCHLIST_CACHE"]))

        reset_watch_state()  # a new process: only the file remains
        with mock.patch.object(pixel_extractor, "_fetch_watchdog",
                               side_effect=RuntimeError("watchdog down")):
            df = pixel_extractor.get_watch_list("x:1")
        self.assertEqual(list(df.source_name), ["sun", "Cyg A"])
        self.assertEqual(list(df.columns), pixel_extractor.WATCH_COLUMNS)

    def test_last_good_list_in_memory(self):
        with mock.patch.object(pixel_extractor, "_fetch_watchdog",
                               return_value=watch_df(["sun"])):
            pixel_extractor.get_watch_list("x:1")
        os.remove(os.environ["EPIC_WATCHLIST_CACHE"])
        with mock.patch.object(pixel_extractor, "_fetch_watchdog",
                               side_effect=OSError("unreachable")):
            df = pixel_extractor.get_watch_list("x:1")
        self.assertEqual(list(df.source_name), ["sun"])

    def test_watchdog_not_asked_again_at_once_after_a_failure(self):
        with mock.patch.object(pixel_extractor, "_fetch_watchdog",
                               side_effect=RuntimeError("down")) as f:
            pixel_extractor.get_watch_list("x:1")
            pixel_extractor.get_watch_list("x:1")
            self.assertEqual(f.call_count, 1)
            pixel_extractor._watch["grpc_retry_at"] = 0.0
            pixel_extractor.get_watch_list("x:1")
            self.assertEqual(f.call_count, 2)

    def test_no_list_means_no_pixels_and_no_error(self):
        with mock.patch.object(pixel_extractor, "_fetch_watchdog",
                               side_effect=RuntimeError("down")), \
             mock.patch.object(pixel_extractor, "get_image_headers_sv",
                               side_effect=AssertionError("not reached")):
            df = pixel_extractor.get_watch_list("x:1")
            self.assertEqual(len(df), 0)
            reset_watch_state()
            idx = pixel_extractor.get_pixel_indices(0, 128, 1.0, 10.0, "x:1")
        self.assertEqual(int(idx["nsrc"][0]), 0)
        self.assertEqual(len(idx["pix_x"]), 0)

    def test_corrupt_cache_is_ignored(self):
        with open(os.environ["EPIC_WATCHLIST_CACHE"], "w") as f:
            f.write("{not json")
        with mock.patch.object(pixel_extractor, "_fetch_watchdog",
                               side_effect=RuntimeError("down")):
            df = pixel_extractor.get_watch_list("x:1")
        self.assertEqual(len(df), 0)

    def test_digest_errors_never_raise(self):
        with mock.patch.object(daily_digest, "_digest_rows",
                               side_effect=ValueError("bad header")):
            self.assertEqual(daily_digest.ingest_daily_digest({}, {}, None, 0.0), 0)


@unittest.skipIf(PGBIN is None, "PostgreSQL binaries not installed")
class WithDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pg = PgServer()
        cls.pg.psql(WATCHDOG_DDL)
        cls.pg.psql(DIGEST_DDL)

    @classmethod
    def tearDownClass(cls):
        cls.pg.close()

    def setUp(self):
        reset_watch_state()
        daily_digest._state.update(engine=None, log_at={}, dropped=0)
        daily_digest._state["pending"].clear()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        reset_watch_state()
        daily_digest._state["engine"] = None

    def test_watch_list_from_the_head_database(self):
        self.pg.psql("""
            TRUNCATE public.epic_watchdog;
            INSERT INTO public.epic_watchdog (source, event_time, ra_deg, dec_deg, event_type,
                t_start, t_end, reason, author, watch_status, patch_type) VALUES
              ('sun', now(), 0, 0, 'body', now() - interval '1 year', now() + interval '99 years',
               'test', 'test', 'watching', 5),
              ('expired', now(), 1, 1, 'frb', now() - interval '2 hours', now() - interval '1 hour',
               'test', 'test', 'watching', 5),
              ('done', now(), 2, 2, 'frb', now() - interval '2 hours', now() + interval '1 hour',
               'test', 'test', 'watched', 5),
              ('later', now(), 3, 3, 'frb', now() + interval '1 hour', now() + interval '2 hours',
               'test', 'test', 'watching', 5),
              ('CHIME-FRB#1', now(), 123.25, -4.5, 'frb', now() - interval '5 min',
               now() + interval '1 hour', 'test', 'test', 'watching', 7);
        """)
        env = {"EPIC_WATCH_DSN": self.pg.dsn(), "EPIC_WATCHLIST_CACHE": ""}
        with mock.patch.dict(os.environ, env), \
             mock.patch.object(pixel_extractor, "_fetch_watchdog",
                               side_effect=RuntimeError("watchdog down")):
            df = pixel_extractor.get_watch_list("x:1")
        self.assertEqual(list(df.source_name), ["sun", "CHIME-FRB#1"])
        self.assertEqual(list(df.columns), pixel_extractor.WATCH_COLUMNS)
        row = df[df.source_name == "CHIME-FRB#1"].iloc[0]
        self.assertAlmostEqual(row.ra, 123.25)
        self.assertEqual(int(row.kernel_dim), 7)

    def test_head_database_down_falls_back_to_the_cache(self):
        cache = os.path.join(self.tmp, "w.json")
        dead = f"host=127.0.0.1 port={free_port()} dbname=epic user=postgres"
        with mock.patch.dict(os.environ, {"EPIC_WATCH_DSN": dead, "EPIC_WATCHLIST_CACHE": cache}):
            with mock.patch.object(pixel_extractor, "_fetch_watchdog",
                                   return_value=watch_df(["sun"])):
                pixel_extractor.get_watch_list("x:1")
            reset_watch_state()
            t0 = time.monotonic()
            with mock.patch.object(pixel_extractor, "_fetch_watchdog",
                                   side_effect=RuntimeError("down")):
                df = pixel_extractor.get_watch_list("x:1")
            self.assertLess(time.monotonic() - t0, 6.0)
        self.assertEqual(list(df.source_name), ["sun"])

    def _count(self):
        return int(self.pg.psql("SELECT count(*) FROM sevilleta.epic_daily_digest"))

    def test_digest_rows_wait_while_the_database_is_down(self):
        self.pg.psql("TRUNCATE sevilleta.epic_daily_digest")
        rows = [dict(source_name="sun", img_time=f"2026-10-17T12:00:{i:02d}",
                     stokes_i=[1.0, 2.0], stokes_v=[0.1, 0.2], cfreq=17.6e6) for i in range(5)]
        dead = f"host=127.0.0.1 port={free_port()} dbname=epic user=postgres"
        with mock.patch.dict(os.environ, {"EPIC_NODE_DSN": dead}):
            t0 = time.monotonic()
            self.assertEqual(daily_digest.write_digest_rows(rows[:3]), 0)
            self.assertEqual(daily_digest.write_digest_rows(rows[3:4]), 0)
            self.assertLess(time.monotonic() - t0, 12.0)
        self.assertEqual(len(daily_digest._state["pending"]), 4)

        daily_digest._state["engine"] = None  # the database is back
        with mock.patch.dict(os.environ, {"EPIC_NODE_DSN": self.pg.dsn()}):
            self.assertEqual(daily_digest.write_digest_rows(rows[4:]), 5)
        self.assertEqual(len(daily_digest._state["pending"]), 0)
        self.assertEqual(self._count(), 5)

    def test_digest_server_restart(self):
        self.pg.psql("TRUNCATE sevilleta.epic_daily_digest")
        row = dict(source_name="sun", img_time="2026-10-17T13:00:00",
                   stokes_i=[1.0], stokes_v=[0.0], cfreq=17.6e6)
        with mock.patch.dict(os.environ, {"EPIC_NODE_DSN": self.pg.dsn()}):
            self.assertEqual(daily_digest.write_digest_rows([row]), 1)
            self.pg.stop()
            try:
                self.assertEqual(daily_digest.write_digest_rows([row]), 0)
            finally:
                self.pg.start()
            # the pooled connection is stale: pool_pre_ping replaces it
            self.assertEqual(daily_digest.write_digest_rows([]), 1)
        self.assertEqual(self._count(), 2)

    def test_ingest_daily_digest_sums_the_patch(self):
        self.pg.psql("TRUNCATE sevilleta.epic_daily_digest")
        nchan, n = 4, 8
        data = np.zeros((4, nchan, n, n), dtype=np.float32)
        data[0, :, 2:4, 2:4] = 1.0   # XX over a 2x2 patch
        data[1, :, 2:4, 2:4] = 2.0   # YY
        data[3, :, 2:4, 2:4] = -0.5  # Im(XY)
        idx = dict(nsrc=np.array([1]), ncoords=np.array([4]),
                   src_ids=np.array(["sun"] * 4),
                   pix_x=np.array([3, 4, 3, 4]), pix_y=np.array([3, 3, 4, 4]))
        with mock.patch.dict(os.environ, {"EPIC_NODE_DSN": self.pg.dsn()}), \
             mock.patch.object(daily_digest, "get_pixel_indices2", return_value=idx):
            n_written = daily_digest.ingest_daily_digest(
                {"DATETIME": "2026-10-17T14:00:00.000"}, {}, data, 17.6e6)
        self.assertEqual(n_written, 1)
        got = self.pg.psql("SELECT source_name, stokes_i, cfreq FROM sevilleta.epic_daily_digest")
        name, stokes_i, cfreq = got.split("|")
        self.assertEqual(name, "sun")
        self.assertEqual(json.loads(stokes_i.replace("{", "[").replace("}", "]")), [12.0] * nchan)
        self.assertAlmostEqual(float(cfreq), 17.6e6, delta=10)


if __name__ == "__main__":
    unittest.main()
