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
        # without the window's noise the error is ERR_FACTOR x fluence / S/N
        self.assertAlmostEqual(out["fluence_err_jy_ms"], FX.ERR_FACTOR * expect / 20, places=2)
        # with it, the error follows from the noise of each channel's contribution
        out = FX.candidate_flux(np.ones(64), np.full(64, 2.0), 2, 0.1, 0.4, k, rec, 90.0, 0.0, f,
                                snr=20, contrib_err=np.full(64, 0.5))
        self.assertAlmostEqual(out["fluence_err_jy_ms"], expect * 0.5 / 8.0, places=2)
        self.assertEqual(out["fluence_err_method"], "window noise")
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

    def test_peak_fit_recovers_the_beam_peak(self):
        """A Gaussian of the beam's width through a pixel and its neighbours gives the peak"""
        sig = 1.22
        xy = np.array([[0.0, 0.0]] + [[0.87 * np.cos(a), 0.87 * np.sin(a)]
                                      for a in np.arange(8) * np.pi / 4])
        src = np.array([0.4, -0.3])
        v = 5.0 * np.exp(-np.sum((xy - src) ** 2, axis=1) / (2 * sig ** 2))
        out = FX.peak_fit(xy, v, sig)
        self.assertTrue(out["applied"], out)
        self.assertAlmostEqual(out["factor"] * v[0], 5.0, places=3)
        self.assertAlmostEqual(out["x"], 0.4, places=3)
        self.assertAlmostEqual(out["y"], -0.3, places=3)
        # missing neighbours, or no signal: not applied, factor 1
        self.assertFalse(FX.peak_fit(xy[:5], v[:5], sig)["applied"])
        self.assertEqual(FX.peak_fit(xy, -v, sig)["factor"], 1.0)
        # per channel the correction grows as f^2 (the beam narrows as 1 / f)
        pf = FX.peak_factors(1.1, np.array([50.0, 60.0, 70.0]))
        self.assertAlmostEqual(pf[1], 1.1 ** (60 / 60.0) ** 2, places=6)
        self.assertGreater(pf[2], pf[1])
        self.assertGreater(pf[1], pf[0])

    def test_high_pass_is_undone_in_the_contributions(self):
        """channel_contributions with a_hp inverts the tiers' high-pass inside each span"""
        from skymon import kernels
        from skymon.quality import channel_contributions

        e = Band(2336, 16).edges_mhz()
        C, W, a = 16, 64, 1.0 / 32
        x = np.zeros((C, W), np.float32)
        row, t = 10, 50
        inv2 = e ** -2
        lag_e = row * (inv2[0] - inv2) / (inv2[0] - inv2[-1])
        for c in range(C):  # one sample per channel, at its upper edge's arrival
            x[c, t - int(np.floor(lag_e[c + 1]))] = 3.0
        # the tier's high-pass, sample by sample (skymon_tier_push)
        hist = np.zeros((C, W, 1), np.float32)
        base = np.zeros((C, 1), np.float32)
        for k in range(W):
            kernels.tier_push(np.ascontiguousarray(x[:, k:k + 1]), base, hist, k, a)
        y = hist[:, :, 0]
        raw = channel_contributions(x, e, row, t, 1)
        lost = channel_contributions(y, e, row, t, 1)
        fixed = channel_contributions(y, e, row, t, 1, a_hp=a)
        self.assertLess(lost.sum(), 0.97 * raw.sum())
        np.testing.assert_allclose(fixed, raw, rtol=1e-5)

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
        from skymon.candidates import separation_deg
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
        # a bright burst (beyond the 12 sigma clip in single cells), 40 s earlier,
        # high and away from Cyg A and the first burst
        rb = next(float(r) for r in np.arange(0, 360, 2.0)
                  if altaz(r, 50.0, T0 + 110)[0] > 70
                  and separation_deg(r, 50.0, 299.868, 40.734) > 20
                  and separation_deg(r, 50.0, ra, dec) > 20)
        cls.bright = dict(id="frb-bright", ra=rb, dec=50.0, dm=50.0, snr=120.0, width_s=0.005,
                          t_inf=T0 + 110.0 - K_DM * 50.0 / f_top ** 2)
        cls.calinfo = os.path.join(cls.out, "calinfo.json")
        sim = SkySimulator(chan0=2336, nchan=64, start_unix=T0, frb_per_hour=0, rfi_per_hour=0,
                           rfi_channel=False, seed=11, frb_list=[cls.frb, cls.bright], el_min=40.0,
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
        # image units are the simulator's flux units: with the pixel's response
        # to the calibrator's offset and a quadratic ring background, k is 1
        # (0.932 x 1.053 before, two errors cancelling by chance)
        self.assertEqual(cyg["background"], "quadratic")
        self.assertTrue(0.85 < cyg["pixel_response"] <= 1.01, cyg["pixel_response"])
        k = np.median(np.array(m["k"], float))
        self.assertTrue(0.97 < k < 1.03, k)
        st = self.mon.status()["calibration"]
        self.assertEqual(st["records"]["latest"]["gain_version"], 0)
        self.assertIsNotNone(st["flux"])

    def _best(self, frb):
        from skymon.candidates import separation_deg

        near = [c for c in self.cands if separation_deg(frb["ra"], frb["dec"], c["ra"], c["dec"])
                < 2 * self.mon.beam and c["cls"] in ("candidate", "known")]
        self.assertTrue(near, [(c["cls"], c["snr"], c["dm"], c["ra"], c["dec"]) for c in self.cands])
        best = max(near, key=lambda c: c["snr"])
        self.assertIsNotNone(best["flux"], best["id"])
        truth = next(t for t in self.sim.truth if t["id"] == frb["id"])["meta"]["fluence_jy_ms"]
        return best, truth

    def _check_fluence(self, best, truth):
        """The fluence within 3 of its errors and 3% (the expected systematic residual is 1%)"""
        fl = best["flux"]
        ratio = fl["fluence_jy_ms"] / truth
        err = fl["fluence_err_jy_ms"] / truth
        self.assertEqual(fl["fluence_err_method"], "window noise")
        self.assertTrue(0.0 < err < 0.15, (err, fl))
        self.assertLess(abs(ratio - 1.0), 3 * err + 0.03, (ratio, err, fl["fluence_jy_ms"], truth,
                                                          best["snr"], fl.get("peak_correction")))
        return ratio, err

    def test_fluence_of_the_burst(self):
        best, truth = self._best(self.frb)
        self._check_fluence(best, truth)
        pk = best["flux"]["peak_correction"]
        self.assertTrue(pk["applied"], pk)
        self.assertTrue(1.0 <= pk["factor"] < 1.3, pk)
        # the products can redo the measurement: raw noise, tier and the calibration record
        npz = np.load(best["files"]["npz"])
        self.assertEqual(npz["raw_sigma"].shape, (64,))
        self.assertEqual(int(npz["tier_images"]), 2 ** best["tier"])
        self.assertEqual(json.loads(str(npz["calinfo"]))["solver"], "skymon simulator")

    def test_bright_burst_survives_the_clip(self):
        """
        At S/N 120 single cells of the burst exceed the 12 sigma clip. They are
        kept because their sky neighbours deviate with them (before, the core
        pixels were zeroed: the burst was found 1.5 deg away at S/N 54 with 0.45
        of its fluence).
        """
        from skymon.candidates import separation_deg

        best, truth = self._best(self.bright)
        self.assertGreater(best["snr"], 90)
        self.assertLess(separation_deg(self.bright["ra"], self.bright["dec"], best["ra"], best["dec"]),
                        self.mon.grid.pix_deg)
        pk = best["flux"]["peak_correction"]
        self.assertTrue(pk["applied"], pk)
        self.assertLess(separation_deg(self.bright["ra"], self.bright["dec"], pk["ra_deg"], pk["dec_deg"]),
                        0.3)
        self._check_fluence(best, truth)

if __name__ == "__main__":
    unittest.main()
