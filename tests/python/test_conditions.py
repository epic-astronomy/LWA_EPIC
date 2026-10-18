"""
Tests of the data-quality periods ("conditions", src/python/skymon/conditions.py):
the samples of a monitor, their HTTP interface and database table, the
conditions carried by candidates, and their summary in the combiner's events.

    cd tests/python && python -m unittest -v test_conditions

A simulated sky (nside 16, 64 channels) is searched for a few minutes, with
the noise raised threefold for one minute in the middle.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src", "python"))

T0 = 1792008000.0
NOISY = (180.0, 240.0)  # seconds after the start with three times the noise


def simulate(out, duration, noisy=None, db_dsn=None, every=None, seed=7):
    """
    Search a simulated sky for duration seconds; during noisy (seconds from
    the start) the noise is three times its normal level. every(i, mon) is
    called after each image.
    """
    from skymon.geometry import Band
    from skymon.monitor import Config, Monitor
    from skymon.simulate import SkySimulator, tier_dt_function

    sim = SkySimulator(chan0=2336, nchan=64, start_unix=T0, frb_per_hour=0, rfi_per_hour=0,
                       seed=seed, el_min=25.0,
                       tier_dt=tier_dt_function(Band(2336, 64).edges_mhz(), 0.083592, 64, 300.0,
                                                64))
    mon = Monitor(Config(out_dir=out, db_dsn=db_dsn, nside=16, dm_max=300.0, block0=64,
                         nthreads=4, warmup_images=32, inject_interval_s=0, seed=seed,
                         accum_hours=0.2))
    t_start = sim._t(0)
    for i in range(int(duration / sim.dt)):
        meta, img = sim.next_image()
        if noisy and noisy[0] <= meta.unix - t_start < noisy[1]:
            # independent noise of the same kind with 8 times the variance:
            # the total is 3 times the normal noise level
            extra = sim._noise(img.shape[0], img.shape[1])
            extra *= sim.sigma * np.float32(np.sqrt(8.0))
            img += extra
        mon.process(meta, img)
        if every is not None:
            every(i, mon)
    mon.store.flush()
    return sim, mon, t_start


class TestConditions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = tempfile.mkdtemp()
        cls.sim, cls.mon, cls.t0 = simulate(cls.out, 330, noisy=NOISY)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)

    def window(self, a, b):
        """Samples whose interval ends in (t0 + a, t0 + b]"""
        return [s for s in self.mon.cond.samples() if self.t0 + a < s["t"] <= self.t0 + b]

    def test_samples_on_the_image_clock(self):
        s = self.mon.cond.samples()
        self.assertGreaterEqual(len(s), 30)
        dt = np.diff([r["t"] for r in s])
        np.testing.assert_allclose(dt, 10.0)
        full = s[1:]  # the first interval starts with the first image
        self.assertTrue(all(118 <= r["images"] <= 121 for r in full), [r["images"] for r in full])
        self.assertTrue(all(r["missing_fraction"] == 0 for r in full))

    def test_noise_raised_for_a_minute(self):
        quiet = self.window(60, NOISY[0])
        self.assertGreaterEqual(len(quiet), 10)
        for r in quiet:
            self.assertLess(r["noise_ratio"], 1.3, r)
            self.assertFalse(r["disturbed"], r)
        # the noise ratio follows the running variance (5 min), so it passes
        # 1.5 some 20-30 s into the noisy minute; the trigger rate at once
        noisy = self.window(NOISY[0] + 10, NOISY[1])
        self.assertTrue(all(r["disturbed"] for r in noisy), noisy)
        late = self.window(NOISY[0] + 40, NOISY[1] + 10)
        self.assertTrue(late and all(r["noise_ratio"] > 1.5 for r in late),
                        [r["noise_ratio"] for r in late])
        self.assertIn("noise", late[-1]["reasons"])
        st = self.mon.status()["conditions"]
        self.assertGreater(st["summary"]["max_noise_ratio"], 1.5)
        self.assertTrue(st["summary"]["disturbed"])
        self.assertIsNotNone(st["summary"]["last_disturbed"])
        self.assertEqual(st["latest"]["t"], self.mon.cond.samples()[-1]["t"])

    def test_search_noise_at_fixed_widths(self):
        """Each sample records the search noise against white noise at about 1, 7 and 20 s"""
        s = self.mon.cond.samples()
        for r in s:
            self.assertEqual([x["target_s"] for x in r["search_noise"]], [1.0, 7.0, 20.0])
            self.assertTrue(-90 <= r["el_cyg_a"] <= 90 and -90 <= r["el_cas_a"] <= 90, r)
        quiet = [x for r in self.window(120, NOISY[0]) for x in r["search_noise"]
                 if x["target_s"] == 1.0]
        self.assertTrue(quiet and all(x["ratio"] is not None for x in quiet), quiet)
        for x in quiet:  # a simulated sky of radiometer noise: about white
            self.assertLess(abs(x["ratio"] - 1.0), 0.3, x)
        noisy = [x["ratio"] for r in self.window(NOISY[0] + 40, NOISY[1] + 10)
                 for x in r["search_noise"] if x["target_s"] == 1.0]
        self.assertTrue(noisy and max(noisy) > 1.2 * float(np.median([x["ratio"] for x in quiet])),
                        (noisy, quiet))

    def test_candidates_carry_conditions(self):
        cands = self.mon.store.list()
        self.assertTrue(cands)
        for c in cands:
            self.assertIn("conditions", c.quality, c.id)
        during = [c for c in cands if self.t0 + NOISY[0] + 20 < c.t_bottom <= self.t0 + NOISY[1]]
        self.assertTrue(during, [(c.cls, c.t_bottom - self.t0) for c in cands])
        for c in during:
            q = c.quality["conditions"]
            self.assertGreaterEqual(q["n_samples"], 1)
            self.assertTrue(q["disturbed"], (c.id, q))
        # the record on disk has them too
        with open(during[0].files["json"]) as f:
            self.assertIn("conditions", json.load(f)["quality"])

    def test_http_series(self):
        from skymon.server import serve
        httpd = serve(self.mon, host="127.0.0.1", port=0)
        port = httpd.server_address[1]
        try:
            got = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/conditions"))
            self.assertEqual([s["t"] for s in got["samples"]],
                             [s["t"] for s in self.mon.cond.samples()])
            self.assertEqual(got["thresholds"]["noise_ratio"], 1.5)
            since = self.t0 + NOISY[0]
            got = json.load(urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/conditions?since={since}"))
            self.assertTrue(got["samples"])
            self.assertTrue(all(s["t"] > since for s in got["samples"]))
            self.assertTrue(any(s["disturbed"] for s in got["samples"]))
            st = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status"))
            self.assertIn("latest", st["conditions"])
        finally:
            httpd.shutdown()
            httpd.server_close()


class TestCombinerConditions(unittest.TestCase):
    def test_event_carries_the_bands_conditions(self):
        from test_combiner import BANDS, FakeMonitor, make, zenith_ish

        class Mon(FakeMonitor):
            def __init__(self, *a, cond=None, **kw):
                super().__init__(*a, **kw)
                self.cond = cond

            def get(self, path):
                out = super().get(path)
                if path.startswith("/api/watches"):
                    for w in out:
                        if w.get("result"):
                            w["result"]["conditions"] = self.cond
                return out

        quiet = dict(n_samples=6, max_noise_ratio=1.02, rfi_rate=0.0, disturbed=False)
        storm = dict(n_samples=6, max_noise_ratio=2.4, rfi_rate=5.0, disturbed=True)
        ra, dec = zenith_ish(T0)
        for lo_cond, expect in ((quiet, False), (storm, True)):
            nodes = {n: Mon(n, b, followup_snr=9.0, cond=quiet) for n, b in BANDS.items()}
            nodes["lo"].cond = lo_cond
            nodes["hi"].candidate("c1", 15.0, 300.0, T0, ra, dec)
            nodes["hi"].cands[-1]["quality"] = dict(conditions=quiet)
            comb, out = make(nodes)
            try:
                t = T0 + 300
                for _ in range(2000):
                    comb.poll(now=t)
                    if all(e["status"] != "pending" for e in comb.events.values()):
                        break
                    t += 10.0
                (ev,) = comb.events.values()
                self.assertEqual(ev["status"], "confirmed")
                self.assertEqual(set(ev["conditions"]), {"hi", "mid", "lo"})
                self.assertEqual(ev["conditions"]["lo"]["disturbed"], expect)
                self.assertEqual(ev["disturbed"], expect)
            finally:
                shutil.rmtree(out, ignore_errors=True)


try:
    from test_localdb import PGBIN, PgServer
except ImportError:  # pragma: no cover
    PGBIN = None


@unittest.skipIf(PGBIN is None, "PostgreSQL binaries not installed")
class TestConditionsDatabase(unittest.TestCase):
    def test_samples_reach_the_table_through_an_outage(self):
        pg = PgServer()
        out = tempfile.mkdtemp()
        state = dict(stopped=False, restarted=False, during=None)

        def every(i, mon):
            # the database thread's work, done here at fixed points of the
            # data so the test is deterministic
            if i % 60 == 0:
                mon.sync_db()
            n = len(mon.cond.samples())
            if n == 4 and not state["stopped"]:
                mon.sync_db()
                pg.stop()
                state["stopped"] = True
            if n == 8 and not state["restarted"]:
                state["during"] = len(mon.cond.unwritten)
                pg.start()
                state["restarted"] = True

        try:
            sim, mon, t0 = simulate(out, 110, db_dsn=pg.dsn(), every=every)
            self.assertTrue(state["restarted"])
            # samples taken during the outage waited in memory
            self.assertGreaterEqual(state["during"], 3)
            mon.sync_db()
            self.assertEqual(len(mon.cond.unwritten), 0)
            rows = mon.db.execute("SELECT t, value FROM conditions ORDER BY t", fetch=True)
            self.assertEqual([r[0] for r in rows], [s["t"] for s in mon.cond.samples()])
            self.assertIn("noise_ratio", rows[-1][1])
            st = mon.status()["conditions"]
            self.assertEqual(st["unwritten"], 0)
        finally:
            pg.close()
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
