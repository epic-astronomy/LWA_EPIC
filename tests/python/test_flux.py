"""
Tests of the flux calibration of candidates (src/python/skymon/flux.py): the
beam table, the fluence arithmetic, the calibration records, and an end-to-end
run on a simulated sky whose calibration record (in the simulator's flux
units) lets the monitor measure the fluence of a burst of known fluence.

    cd tests/python && python -m unittest -v test_flux
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src", "python"))

from skymon import flux as FX  # noqa: E402
from skymon.dedisp import K_DM  # noqa: E402
from skymon.geometry import Band, altaz  # noqa: E402


def flat_record(nchan=64, chan0=2336, fx=1.0, beam_fn=lambda alt: np.ones_like(alt)):
    alt = np.arange(0.0, 90.0 + 1e-9, 5.0)
    az = np.arange(0.0, 360.0, 10.0)
    tab = np.repeat(beam_fn(alt)[:, None], len(az), axis=1).tolist()
    band = Band(chan0, nchan)
    return dict(layout=1, gain_version=3, applied=True, t_solved_unix=0.0, chan0=chan0, nchan=nchan,
                solve_chans=list(range(chan0, chan0 + nchan, 4)),
                flux_scale=dict(X=[fx] * nchan, Y=[fx] * nchan), calibrators=[],
                beam=dict(freqs_mhz=[band.edges_mhz()[0], band.f_centre, band.edges_mhz()[-1]],
                          alt_deg=alt.tolist(), az_deg=az.tolist(), X=[tab] * 3, Y=[tab] * 3))


class TestArithmetic(unittest.TestCase):
    def test_beam_table_interpolation(self):
        rec = flat_record(beam_fn=lambda alt: np.sin(np.radians(alt)))
        f = Band(2336, 64).centres_mhz()
        for el in (90.0, 60.0, 32.5, 12.0):
            bx, by = FX.beam(rec["beam"], el, 123.0, f)
            np.testing.assert_allclose(bx, np.sin(np.radians(el)), atol=0.01)
            np.testing.assert_allclose(by, bx)
        # azimuth wraps
        np.testing.assert_allclose(FX.beam(rec["beam"], 45.0, 355.0, f)[0], FX.beam(rec["beam"], 45.0, 5.0, f)[0])

    def test_fluence_of_a_known_signal(self):
        """sum_track(x) sigma sqrt(2^tier) dt / (k Beff), in Jy ms"""
        f = Band(2336, 64).centres_mhz()
        rec = flat_record(fx=2.0)  # Beff = F^2 B = 4
        k = np.full(64, 0.5)
        out = FX.candidate_flux(np.ones(64), np.full(64, 2.0), 2, 0.1, 0.4, k, rec, 90.0, 0.0, f, snr=20)
        expect = 1.0 * 2.0 * 2.0 * 0.1 / (0.5 * 4.0) * 1e3
        self.assertAlmostEqual(out["fluence_jy_ms"], expect, places=2)
        self.assertAlmostEqual(out["fluence_err_jy_ms"], expect / 20, places=2)
        self.assertAlmostEqual(out["flux_density_jy"], expect / 1e3 / 0.4, places=2)
        self.assertEqual(out["gain_version"], 3)
        # nominal images (no table applied): the absolute scale is 1
        rec["applied"] = False
        out = FX.candidate_flux(np.ones(64), np.full(64, 2.0), 2, 0.1, 0.4, k, rec, 90.0, 0.0, f)
        self.assertAlmostEqual(out["fluence_jy_ms"], 4 * expect, places=1)
        # without a measurement there is no flux
        self.assertIsNone(FX.candidate_flux(np.ones(64), np.ones(64), 0, 0.1, 0.1, None, rec, 90, 0, f))

    def test_lsl_beam_table(self):
        """The monitor's own primary beam: the LWA dipole from lsl, zenith = 1, falling to the horizon"""
        f = Band(2336, 64)
        e = f.edges_mhz()
        tab = FX.beam_table([e[0], f.f_centre, e[-1]])
        c = f.centres_mhz()
        for pol in (0, 1):
            np.testing.assert_allclose(FX.beam(tab, 90.0, 0.0, c)[pol], 1.0, atol=1e-6)
        b45 = np.mean(0.5 * (FX.beam(tab, 45.0, 0.0, c)[0] + FX.beam(tab, 45.0, 0.0, c)[1]))
        b20 = np.mean(FX.beam(tab, 20.0, 0.0, c)[0])
        self.assertTrue(1.0 > b45 > b20 > 0.0, (b45, b20))
        # X and Y differ off the zenith (dipoles at right angles)
        self.assertGreater(abs(FX.beam(tab, 30.0, 0.0, c)[0].mean() - FX.beam(tab, 30.0, 0.0, c)[1].mean()), 1e-3)
        # the monitor's beam is used when the record has none; a record's beam
        # (a calibrated beam) takes precedence
        rec = flat_record()  # flat beam table, zenith = 1 everywhere
        args = (np.ones(64), np.ones(64), 0, 0.1, 0.1, np.ones(64))
        out_rec = FX.candidate_flux(*args, rec, 45.0, 0.0, c, beam_tab=tab)
        self.assertNotEqual(out_rec["beam_model"], "empirical")
        no_beam = dict(rec)
        del no_beam["beam"]
        out_lsl = FX.candidate_flux(*args, no_beam, 45.0, 0.0, c, beam_tab=tab)
        self.assertEqual(out_lsl["beam_model"], "empirical")
        self.assertAlmostEqual(out_lsl["fluence_jy_ms"] / out_rec["fluence_jy_ms"], 1 / b45, delta=0.02 / b45)

    def test_feed_reads_new_records_only(self):
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "calinfo.json")
            feed = FX.CalibrationFeed(p)
            self.assertIsNone(feed.poll())  # no file yet
            rec = flat_record()
            with open(p, "w") as f:
                json.dump(rec, f)
            self.assertEqual(feed.poll()["gain_version"], 3)
            self.assertIsNone(feed.poll())  # unchanged
            rec2 = dict(rec, gain_version=4, t_solved_unix=10.0)
            with open(p + ".tmp", "w") as f:
                json.dump(rec2, f)
                f.write(" " * 7)  # another size, so the change is seen within a clock tick
            os.replace(p + ".tmp", p)
            self.assertEqual(feed.poll()["gain_version"], 4)
            self.assertEqual(feed.for_version(3)["gain_version"], 3)
            self.assertEqual(feed.for_version(0)["gain_version"], 4)  # nominal: the latest
            self.assertEqual(len(feed.unwritten), 2)
        finally:
            shutil.rmtree(d, ignore_errors=True)


def high_time(t0, ra, dec, min_el):
    """The first time from t0 (in steps of 10 min) at which (ra, dec) is above min_el"""
    for k in range(144):
        t = t0 + 600 * k
        if altaz(ra, dec, t)[0] > min_el:
            return t
    raise AssertionError("never high enough")


class TestEndToEnd(unittest.TestCase):
    """A burst of known fluence on a simulated sky, calibrated on its point sources"""

    @classmethod
    def setUpClass(cls):
        from skymon.candidates import separation_deg  # noqa: F401
        from skymon.monitor import Config, Monitor
        from skymon.simulate import SkySimulator, tier_dt_function

        cls.out = tempfile.mkdtemp()
        T0 = high_time(1792008000.0, 299.868, 40.734, 55.0)  # Cyg A high
        band = Band(2336, 64)
        # the burst near the zenith, 60 deg from Cyg A
        ra, dec = None, 34.0
        for r in np.arange(0, 360, 2.0):
            if altaz(r, dec, T0 + 150)[0] > 75:
                ra = float(r)
                break
        f_top = band.f_hi
        cls.frb = dict(id="frb-flux", ra=ra, dec=dec, dm=50.0, snr=40.0, width_s=0.005,
                       t_inf=T0 + 150.0 - K_DM * 50.0 / f_top ** 2)
        cls.calinfo = os.path.join(cls.out, "calinfo.json")
        sim = SkySimulator(chan0=2336, nchan=64, start_unix=T0, frb_per_hour=0, rfi_per_hour=0,
                           rfi_channel=False, seed=11, frb_list=[cls.frb], el_min=40.0,
                           calinfo_path=cls.calinfo,
                           tier_dt=tier_dt_function(band.edges_mhz(), 0.083592, 64, 300.0, 64))
        cls.sim = sim
        cls.mon = Monitor(Config(out_dir=cls.out, nside=64, el_min=40.0, dm_max=300.0, block0=64,
                                 nthreads=4, warmup_images=32, inject_interval_s=0, seed=11,
                                 accum_hours=0.2, calinfo_path=cls.calinfo,
                                 beam_model="record"))
        for _ in range(int(200 / sim.dt)):
            meta, img = sim.next_image()
            cls.mon.process(meta, img)
        cls.mon.store.flush()
        cls.cands = [c.to_dict() for c in cls.mon.store.list()]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)

    def test_calibrators_measured(self):
        m = self.mon.flux_meas
        self.assertIsNotNone(m)
        names = [c["name"] for c in m["calibrators"]]
        self.assertIn("CygA", names, m)
        cyg = next(c for c in m["calibrators"] if c["name"] == "CygA")
        self.assertGreater(cyg["excess_snr"], 20)
        # image units are the simulator's flux units: k is about 1 (less where the
        # HEALPix pixel misses the source's peak)
        k = np.median(np.array(m["k"], float))
        self.assertTrue(0.7 < k < 1.1, k)
        st = self.mon.status()["calibration"]
        self.assertEqual(st["records"]["latest"]["gain_version"], 0)
        self.assertIsNotNone(st["flux"])

    def test_fluence_of_the_burst(self):
        from skymon.candidates import separation_deg

        near = [c for c in self.cands if separation_deg(self.frb["ra"], self.frb["dec"], c["ra"], c["dec"])
                < 2 * self.mon.beam and c["cls"] == "candidate"]
        self.assertTrue(near, [(c["cls"], c["snr"], c["dm"]) for c in self.cands])
        best = max(near, key=lambda c: c["snr"])
        self.assertIsNotNone(best["flux"], best["id"])
        truth = next(t for t in self.sim.truth if t["id"] == "frb-flux")["meta"]["fluence_jy_ms"]
        ratio = best["flux"]["fluence_jy_ms"] / truth
        self.assertTrue(0.6 < ratio < 1.4, (ratio, best["flux"]["fluence_jy_ms"], truth, best["snr"]))
        # the products can redo the measurement: raw noise, tier and the calibration record
        npz = np.load(best["files"]["npz"])
        self.assertEqual(npz["raw_sigma"].shape, (64,))
        self.assertEqual(int(npz["tier_images"]), 2 ** best["tier"])
        self.assertEqual(json.loads(str(npz["calinfo"]))["solver"], "skymon simulator")


if __name__ == "__main__":
    unittest.main()
