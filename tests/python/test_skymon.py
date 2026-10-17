"""
Tests of the all-sky monitor (src/python/skymon).

    cd tests/python && python -m unittest -v test_skymon      (or pytest)

Needs numpy, astropy, healpy, lsl (the imager's Python environment) and gcc.
The end-to-end tests simulate a few minutes of sky at a coarse HEALPix
resolution and take about a minute each.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "python"))

from skymon import kernels  # noqa: E402
from skymon.burst import Burst, channel_fractions  # noqa: E402
from skymon.dedisp import K_DM, brute_force, fdmt, fdmt_plan, make_tiers  # noqa: E402
from skymon.geometry import Band, SkyGrid, bilinear, image_wcs, time_to_set  # noqa: E402
from skymon.shm import RingReader, RingWriter  # noqa: E402

T0 = 1792008000.0  # 2026-10-15 UTC, a fixed epoch for reproducible geometry
BAND = Band(chan0=2336, nchan=64, nbin=2)  # 55.87-58.94 MHz in 64 channels


def dispersed(edges, dt, n_t, dm, t_top, amp=1.0):
    """[C, n_t] dynamic spectrum of a burst reaching the top of the band at t_top"""
    b = Burst("t", 0, 0, dm, t_top - K_DM * dm / edges[-1] ** 2, width_s=0.0)
    return np.stack([channel_fractions(b, edges, k * dt, dt) for k in range(n_t)], 1) * amp


class TestDedispersion(unittest.TestCase):
    def setUp(self):
        self.edges = BAND.edges_mhz()
        self.dt = 0.0836

    def test_kernel_matches_numpy(self):
        plan = fdmt_plan(self.edges, 64)
        x = np.random.default_rng(1).standard_normal((64, 200, 7)).astype(np.float32)
        ref = fdmt(x, plan)
        got = kernels.fdmt(x, kernels.flatten_plan(plan), 64, nthreads=3, ptile=4)
        np.testing.assert_array_equal(ref, got)

    def test_pulse_peaks_at_its_delay(self):
        D = 64
        plan = fdmt_plan(self.edges, D)
        s = K_DM * (self.edges[0] ** -2 - self.edges[-1] ** -2)
        for delay in (5, 20, 47, 63):
            dm = delay * self.dt / s
            dyn = dispersed(self.edges, self.dt, 160, dm, t_top=40 * self.dt, amp=1.0)
            out = fdmt(dyn[:, :, None].astype(np.float32), plan)[:, :, 0]
            d, t = np.unravel_index(np.argmax(out), out.shape)
            self.assertLessEqual(abs(d - delay), 1, (delay, d))
            # all channels' fluence is summed at the peak (within the boxcar of one row)
            self.assertGreater(out[d].max(), 0.6 * 64)

    def test_brute_force_agrees_on_pulses(self):
        D = 48
        plan = fdmt_plan(self.edges, D)
        s = K_DM * (self.edges[0] ** -2 - self.edges[-1] ** -2)
        dm = 30 * self.dt / s
        dyn = dispersed(self.edges, self.dt, 140, dm, 30 * self.dt)[:, :, None]
        a = fdmt(dyn.astype(np.float32), plan)[:, :, 0]
        b = brute_force(dyn, self.edges, D)[:, :, 0]
        self.assertLessEqual(abs(np.argmax(a.max(1)) - np.argmax(b.max(1))), 1)

    def test_tiers_cover_the_dm_range_in_octaves(self):
        tiers = make_tiers(self.edges[0], self.edges[-1], self.dt, 64, 3000.0)
        self.assertGreaterEqual(tiers[-1].dm_max, 3000.0)
        for a, b in zip(tiers, tiers[1:]):
            self.assertAlmostEqual(b.dm_per_delay, 2 * a.dm_per_delay)
            self.assertLessEqual(b.dm_min, a.dm_max + b.dm_per_delay)
            self.assertEqual(b.decim, 2 * a.decim)


class TestSharedMemory(unittest.TestCase):
    def test_roundtrip_overrun_and_restart(self):
        name = f"skymon_test_{os.getpid()}"
        w = RingWriter(name, 4, 3, 8)
        r = RingReader(name)
        try:
            self.assertIsNone(r.read())
            img = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
            w.write(img, 1000, 2000, 2336)
            meta, _ = r.read()  # the reader starts with the newest image
            self.assertEqual(meta.seq_start, 1000)
            self.assertIsNone(r.read())
            w.write(img + 1, 3000, 2000, 2336)
            meta, data = r.read()
            self.assertEqual(meta.seq_start, 3000)
            np.testing.assert_array_equal(data, img + 1)
            for k in range(10):  # overrun: the reader skips and counts
                w.write(img + k, 5000 + 2000 * k, 2000, 2336)
            seen = []
            while (item := r.read()) is not None:
                seen.append(item[0].seq_start)
            self.assertGreater(r.dropped, 0)
            self.assertEqual(seen[-1], 5000 + 2000 * 9)
            w.close()
            w = RingWriter(name, 4, 3, 8)  # writer restart: new session
            w.write(img, 99000, 2000, 2336)
            w.write(img, 101000, 2000, 2336)
            meta, _ = r.read()
            self.assertEqual(r.restarts, 1)
            self.assertEqual(meta.seq_start, 101000)
        finally:
            r.close()
            w.close(unlink=True)

    def test_cpp_publisher_matches_fits_order(self):
        gxx = shutil.which("g++")
        if not gxx:
            self.skipTest("g++ not available")
        tmp = tempfile.mkdtemp()
        exe = os.path.join(tmp, "spw")
        subprocess.run([gxx, "-std=c++17", "-O2", "-o", exe,
                        os.path.join(ROOT, "tests", "cpp", "sky_publisher_writer.cpp")],
                       check=True)
        name = f"skymon_cpp_{os.getpid()}"
        nchan, grid, nbin = 4, 8, 2
        r = RingReader(name)
        try:
            subprocess.run([exe, name, str(nchan), str(grid), str(nbin), "1"], check=True,
                           capture_output=True)
            r.next, r.session = 0, None
            r._open()
            r.session = r._header()[1]
            meta, data = r.read()
            self.assertEqual((meta.nchan, meta.nbin, meta.grid, meta.chan0, meta.nseqs),
                             (nchan // nbin, nbin, grid, 2336, 2000))
            self.assertEqual(meta.gain_version, 7)
            # the imager's buffer and the transform of epic_utils.save_output
            c, a, b, k = np.meshgrid(np.arange(nchan), np.arange(grid), np.arange(grid),
                                     np.arange(4), indexing="ij")
            buf = (c * 1e4 + a * 100 + b + 0.1 * k).astype(np.float32)
            img = np.transpose(buf, (3, 0, 2, 1))
            img = np.fft.fftshift(img, axes=(2, 3))[:, :, ::-1, :]
            stokes_i = img[0] + img[1]
            ref = stokes_i.reshape(nchan // nbin, nbin, grid, grid).sum(1)
            np.testing.assert_allclose(data, ref, rtol=1e-6)
        finally:
            r.close()
            try:
                os.unlink(f"/dev/shm/{name}")
            except FileNotFoundError:
                pass
            shutil.rmtree(tmp, ignore_errors=True)


class TestGeometry(unittest.TestCase):
    def test_bilinear_weights(self):
        idx, w = bilinear(np.array([3.25]), np.array([5.5]), 8)
        np.testing.assert_allclose(w.sum(), 1.0)
        img = np.arange(64, dtype=float).reshape(8, 8)  # value = 8 y0 + x0
        val = float((img.ravel()[idx[0]] * w[0]).sum())
        self.assertAlmostEqual(val, 8 * 4.5 + 2.25, places=5)

    def test_bicubic_weights(self):
        from skymon.geometry import bicubic
        idx, w = bicubic(np.array([4.25]), np.array([5.5]), 12)
        np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-6)
        img = np.add.outer(8.0 * np.arange(12), np.arange(12))  # linear: 8 y0 + x0
        val = float((img.ravel()[idx[0]] * w[0]).sum())
        # Keys a = -0.75 keeps a beam's peak (below) at the cost of exactness on a
        # plane: a gradient g per pixel is offset by at most 0.048 g
        self.assertAlmostEqual(val, 8 * 4.5 + 3.25, delta=0.048 * 9)
        # the peak of a Gaussian beam of three image pixels per FWHM, between pixels
        yy, xx = np.mgrid[0:12, 0:12]
        sig = 1.22
        for x0, y0 in ((4.25, 5.5), (4.5, 5.5), (4.0, 5.0)):
            beam = np.exp(-((xx - (x0 - 1)) ** 2 + (yy - (y0 - 1)) ** 2) / (2 * sig ** 2))
            idx, w = bicubic(np.array([x0]), np.array([y0]), 12)
            peak = float((beam.ravel()[idx[0]] * w[0]).sum())
            self.assertTrue(0.999 < peak < 1.009, (x0, y0, peak))

    def test_tracked_pixels_follow_the_sky(self):
        g = SkyGrid(8, el_min_deg=10.0)
        added, _ = g.update(T0)
        self.assertGreater(len(added), 100)
        live = g.active()
        el = None
        from skymon.geometry import altaz
        el, _ = altaz(g.ra[g.hpix_of[live]], g.dec[g.hpix_of[live]], T0)
        self.assertGreater(el.min(), 10.0)
        added2, removed2 = g.update(T0 + 3600)  # an hour later the sky has turned
        self.assertGreater(len(added2), 0)
        self.assertGreater(len(removed2), 0)
        for s in removed2:
            self.assertEqual(g.slot_of[g.hpix_of[s]] if g.hpix_of[s] >= 0 else s, s)

    def test_source_lands_on_its_pixel(self):
        g = SkyGrid(16, el_min_deg=10.0)
        g.update(T0)
        live = np.nonzero(g.active())[0]
        s = live[len(live) // 3]
        h = g.hpix_of[s]
        wcs = image_wcs(T0, 128, 1.056)
        x, y = wcs.all_world2pix(g.ra[h], g.dec[h], 1, ra_dec_order=True)
        img = np.zeros((1, 128, 128), np.float32)
        yy, xx = np.mgrid[1:129, 1:129]
        img[0] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5 ** 2))
        out = np.zeros((1, g.capacity), np.float32)
        kernels.regrid(img, g.idx, g.w, out)
        self.assertEqual(int(np.argmax(out[0])), s)

    def test_time_to_set(self):
        self.assertTrue(np.isinf(time_to_set(80.0, 0.0, T0, 10.0)))  # circumpolar
        self.assertEqual(float(time_to_set(-70.0, 0.0, T0, 10.0)), 0.0)  # never rises


class TestSearcher(unittest.TestCase):
    def test_detects_injected_burst_and_not_noise(self):
        from skymon.search import Searcher

        rng = np.random.default_rng(3)
        edges = BAND.edges_mhz()
        dt = 0.0836
        P = 64
        se = Searcher(edges, dt, P, dm_max=150.0, block0=64, threshold=7.5, nthreads=2,
                      warmup_images=16, unix0=0.0, rng=rng)
        se.set_live(np.ones(P, bool))
        dm, t_top = 60.0, 150.0
        t_inf = t_top - K_DM * dm / edges[-1] ** 2
        b = Burst("x", 0, 0, dm, t_inf, width_s=0.0)
        n_img = int(300 / dt)
        hits = []
        for n in range(n_img):
            raw = rng.standard_normal((BAND.nchan, P)).astype(np.float32) + 10.0
            frac = channel_fractions(b, edges, n * dt, dt)
            raw[:, 7] += 2.0 * frac
            for res in se.push(raw):
                hits += res.hits
            se.finish_block()
        self.assertTrue(hits, "burst not found")
        best = max(hits, key=lambda h: h.snr)
        self.assertEqual(best.slot, 7)
        self.assertLess(abs(best.dm - dm) / dm, 0.1)
        self.assertLess(abs(best.t_inf - t_inf), 3.0)
        noise = [h for h in hits if h.slot != 7]
        self.assertLessEqual(len(noise), 1)


class TestTrends(unittest.TestCase):
    def test_rising_pixels_give_no_hits(self):
        """A pixel brightening as it rises through the primary beam is not a burst"""
        from skymon.search import Searcher

        rng = np.random.default_rng(4)
        edges = BAND.edges_mhz()
        dt = 0.0836
        P = 48
        se = Searcher(edges, dt, P, dm_max=600.0, block0=64, threshold=8.0, nthreads=2,
                      warmup_images=16, unix0=0.0, rng=rng)
        se.set_live(np.ones(P, bool))
        # up to 0.05 sigma per second in every channel, about four times the
        # brightening of a pixel rising through the primary beam at 10 deg
        slope = np.linspace(0.0, 0.05 * dt, P)[None, :]
        hits = []
        for n in range(int(400 / dt)):
            raw = (rng.standard_normal((BAND.nchan, P)) + 10.0 + slope * n).astype(np.float32)
            for res in se.push(raw):
                hits += res.hits
            se.finish_block()
        self.assertLessEqual(len(hits), 1, [(h.slot, h.tier, round(h.snr, 1)) for h in hits])


class TestSharedFrbList(unittest.TestCase):
    def test_bursts_start_when_they_reach_the_band(self):
        """A high-DM burst listed first must not hold back one of low DM in a low band"""
        from skymon.simulate import SkySimulator
        chan0 = 1442 - 64  # 33.0-36.0 MHz
        f_top = Band(chan0, 128).f_hi
        start = T0
        ra, dec = 0.0, 0.0
        for r in range(0, 360, 5):
            from skymon.geometry import altaz
            if altaz(r, 34.0, start)[0] > 70:
                ra, dec = float(r), 34.0
                break
        high = dict(id="sim-high", ra=ra, dec=dec, dm=500.0, snr=20.0,
                    t_inf=start - K_DM * 500.0 / f_top ** 2 + 400.0)  # reaches the band at +400 s
        low = dict(id="sim-low", ra=ra, dec=dec, dm=50.0, snr=20.0,
                   t_inf=start - K_DM * 50.0 / f_top ** 2 + 5.0)  # reaches it at +5 s
        self.assertLess(high["t_inf"], low["t_inf"])  # listed first by t_inf
        sim = SkySimulator(chan0=chan0, nchan=128, start_unix=start, frb_per_hour=0,
                           rfi_per_hour=0, seed=3, frb_list=[high, low])
        for _ in range(int(10 / sim.dt)):
            sim.next_image()
        self.assertEqual([t["id"] for t in sim.truth], ["sim-low"])


def _sim_and_monitor(out, duration, frb_per_hour, inject_every_s, snr=(25, 40), dm=(10, 120),
                     seed=5):
    from skymon.monitor import Config, Monitor
    from skymon.simulate import SkySimulator, tier_dt_function

    sim = SkySimulator(chan0=2336, nchan=64, start_unix=T0, frb_per_hour=frb_per_hour, snr=snr,
                       dm=dm, rfi_per_hour=20, truth_path=os.path.join(out, "truth.jsonl"),
                       seed=seed, el_min=25.0,
                       tier_dt=tier_dt_function(Band(2336, 64).edges_mhz(), 0.083592, 64, 300.0,
                                                64))
    mon = Monitor(Config(out_dir=out, nside=32, dm_max=300.0, block0=64, nthreads=4,
                         warmup_images=32, inject_interval_s=inject_every_s,
                         inject_snr=(25, 40), inject_dm=(10, 120), seed=seed,
                         accum_hours=0.5))
    for _ in range(int(duration / sim.dt)):
        meta, img = sim.next_image()
        mon.process(meta, img)
    mon.store.flush()
    return sim, mon


class TestEndToEnd(unittest.TestCase):
    def test_simulated_frbs_are_found_and_stored(self):
        from skymon.evaluate import report
        from skymon.server import serve

        out = tempfile.mkdtemp()
        try:
            sim, mon = _sim_and_monitor(out, 420, frb_per_hour=40, inject_every_s=0)
            self.assertGreater(len(sim.truth), 0)
            rep = report(os.path.join(out, "truth.jsonl"), out, t_end=sim._t(sim.n),
                         latency_s=120)
            found = sum(c is not None for _, c in rep["rows"])
            self.assertGreaterEqual(found, max(1, int(0.7 * len(rep["truth"]))), rep["text"])
            self.assertLessEqual(len(rep["false"]), 2, rep["text"])
            # the S/N is not lost to the normalisation, clipping or flagging
            ratio = [c["snr"] / t["snr"] for t, c in rep["rows"] if c is not None]
            self.assertGreater(float(np.median(ratio)), 0.7, rep["text"])
            # nothing that might be a burst is stored without its data
            for c in rep["cands"]:
                if not c.get("rfi_certain"):
                    self.assertIn("npz", c["files"], (c["cls"], c.get("rfi_reasons")))
            # products on disk
            stored = [c for _, c in rep["rows"] if c is not None]
            self.assertTrue(os.path.exists(stored[0]["files"]["npz"]))
            with np.load(stored[0]["files"]["npz"]) as z:
                self.assertEqual(z["dynspec"].shape[0], 64)
            q = stored[0]["quality"]
            self.assertLess(q["snr_dm0_ratio"], 0.8)  # dispersed, not broadband
            self.assertEqual(len(q["subband_snr"]), 4)
            # the HTTP interface
            httpd = serve(mon, host="127.0.0.1", port=0)
            port = httpd.server_address[1]
            try:
                st = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status"))
                self.assertGreater(st["tracked"], 100)
                h = int(mon.grid.hpix_of[np.nonzero(mon.grid.active())[0][0]])
                px = json.load(urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/pixel?hpix={h}"))
                self.assertTrue(px["tracked"])
                self.assertGreater(len(px["times"]), 10)
                self.assertEqual(len(px["data"][0]), 64)
                cl = json.load(urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/candidates"))
                self.assertGreaterEqual(len(cl), found)
            finally:
                httpd.shutdown()
                httpd.server_close()
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_live_injections_are_recovered(self):
        out = tempfile.mkdtemp()
        try:
            sim, mon = _sim_and_monitor(out, 480, frb_per_hour=0, inject_every_s=40)
            mon.injector.expire(math.inf)
            s = mon.injector.stats
            self.assertGreaterEqual(s["injected"], 3)
            self.assertGreaterEqual(s["recovered"], 0.7 * s["injected"], s)
            with open(os.path.join(out, "injections.jsonl")) as f:
                events = [json.loads(line)["event"] for line in f]
            self.assertIn("scheduled", events)
            self.assertIn("recovered", events)
            inj = [c for c in mon.store.list() if c.cls == "injection"]
            self.assertGreaterEqual(len(inj), s["recovered"])
        finally:
            shutil.rmtree(out, ignore_errors=True)


class TestIndependence(unittest.TestCase):
    def test_monitor_survives_missing_and_restarting_imager(self):
        """A monitor waits for its ring, rides out a writer restart and gaps, alone"""
        import threading
        from skymon.__main__ import run_ring
        from skymon.monitor import Config, Monitor
        from skymon.simulate import SkySimulator

        out = tempfile.mkdtemp()
        name = f"skymon_ind_{os.getpid()}"
        mon = Monitor(Config(out_dir=out, nside=8, dm_max=100.0, block0=32, nthreads=2,
                             warmup_images=8, inject_interval_s=0, accum_hours=0.1))
        stop = threading.Event()
        th = threading.Thread(target=run_ring, args=(mon, name, stop),
                              kwargs=dict(status_every=1e9), daemon=True)
        th.start()
        try:
            time.sleep(0.5)  # no ring yet: the monitor just waits
            sim = SkySimulator(chan0=2336, nchan=64, start_unix=T0, frb_per_hour=0,
                               rfi_per_hour=0, seed=2)
            w = RingWriter(name, 16, 64, 128)
            for _ in range(60):
                m, img = sim.next_image()
                w.write(img, m.seq_start, m.nseqs, m.chan0, m.nbin, m.grid_res, m.img_len_ms)
                time.sleep(0.01)
            w.close()
            for _ in range(20):  # a 20-image gap, then the imager restarts
                sim.next_image()
            w = RingWriter(name, 16, 64, 128)
            for _ in range(60):
                m, img = sim.next_image()
                w.write(img, m.seq_start, m.nseqs, m.chan0, m.nbin, m.grid_res, m.img_len_ms)
                time.sleep(0.01)
            deadline = time.time() + 30
            while mon.metrics["images"] < 120 and time.time() < deadline:
                time.sleep(0.1)
            self.assertGreaterEqual(mon.metrics["images"], 120)
            self.assertGreaterEqual(mon.metrics["missing_images"], 1)
            w.close(unlink=True)
        finally:
            stop.set()
            th.join(5)
            shutil.rmtree(out, ignore_errors=True)




class TestDelayFactor(unittest.TestCase):
    def test_twice_the_delays_doubles_each_tiers_dm(self):
        """
        D = 2C: each tier keeps its DM step but reaches twice the DM, so a burst
        is searched in a lower tier (finer time resolution against its
        smearing); fewer tiers reach dm_max
        """
        from skymon.geometry import Band
        from skymon.search import Searcher

        e = Band(2336, 64).edges_mhz()
        one = Searcher(e, 0.0836, 16, dm_max=3000.0, block0=64, nthreads=1).describe()
        two = Searcher(e, 0.0836, 16, dm_max=3000.0, block0=64, nthreads=1, max_delay=128).describe()
        self.assertAlmostEqual(two[0]["dm_step"], one[0]["dm_step"], places=6)
        self.assertAlmostEqual(two[0]["dm_max"], 2 * one[0]["dm_max"], delta=one[0]["dm_step"])
        self.assertGreaterEqual(two[-1]["dm_max"], 3000.0)
        self.assertEqual(len(two), len(one) - 1)

class TestDuplicates(unittest.TestCase):
    """The candidate store keeps distinct bursts and merges only repeats of one burst"""

    def cand(self, cid, snr, ra, dec, t_bottom, dm=120.0):
        from skymon.candidates import Candidate

        return Candidate(id=cid, cls="candidate", snr=snr, dm=dm, dm_step=2.4, width_s=0.17,
                         t_inf=t_bottom - 30.0, t_bottom=t_bottom, t_top=t_bottom - 5.0, utc_inf="",
                         hpix=0, nside=64, ra=ra, dec=dec, el=70.0, az=0.0, image_x=0.0, image_y=0.0,
                         tier=1, tier_dt=0.167, n_pix=3, extent_deg=1.0, f_lo_mhz=55.9,
                         f_hi_mhz=58.9, host="test", detected_unix=time.time())

    def test_distinct_bursts_both_kept_repeat_merged(self):
        from skymon.candidates import CandidateStore

        out = tempfile.mkdtemp()
        try:
            store = CandidateStore(out)
            beam = 3.0
            t0 = 1792008000.0
            self.assertIsNotNone(store.add(self.cand("a", 20.0, 200.0, 34.0, t0), None, beam))
            # a second burst at the same DM, 5 deg and 28 s away: kept
            self.assertIsNotNone(store.add(self.cand("b", 12.0, 206.0, 34.0, t0 + 28.0), None, beam))
            # the first burst found again by another tier, 1 deg off: merged
            self.assertIsNone(store.add(self.cand("a2", 15.0, 201.0, 34.0, t0 + 0.5), None, beam))
            store.flush()
            self.assertEqual(sorted(c.id for c in store.list()), ["a", "b"])
            self.assertEqual(store.counts["duplicates"], 1)
        finally:
            shutil.rmtree(out, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
