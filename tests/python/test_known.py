"""
Tests of known-source labels (skymon/known.py) and of the narrowband
classification (skymon/quality.py, Monitor._candidates).

    cd tests/python && python -m unittest -v test_known

The end-to-end test simulates three minutes of sky at nside 32 with
64 channels: a burst from a pulsar at its DM, and a carrier in one channel.
"""
import math
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "python"))
sys.path.insert(0, HERE)

from skymon.candidates import elevation_radius, separation_deg  # noqa: E402
from skymon.dedisp import K_DM, fdmt, fdmt_plan  # noqa: E402
from skymon.geometry import Band, altaz, beam_fwhm_deg  # noqa: E402
from skymon.known import KnownSources, load_pulsars  # noqa: E402
from skymon.quality import candidate_quality  # noqa: E402
from test_skymon import BAND, T0, dispersed  # noqa: E402


def _sun_low_precision(unix):
    """Astronomical Almanac low-precision solar position (about 0.01 deg), J2000-ish"""
    n = unix / 86400.0 + 2440587.5 - 2451545.0
    L = math.radians((280.460 + 0.9856474 * n) % 360)
    g = math.radians((357.528 + 0.9856003 * n) % 360)
    lam = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2 * g)
    eps = math.radians(23.439 - 4e-7 * n)
    ra = math.degrees(math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))) % 360
    dec = math.degrees(math.asin(math.sin(eps) * math.sin(lam)))
    return ra, dec


class TestKnownSources(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ks = KnownSources(beam_fwhm_deg(BAND.f_centre))

    def test_sun_position(self):
        from astropy.coordinates import get_sun
        from astropy.time import Time

        ra, dec = self.ks.body_positions(T0)["Sun"]
        g = get_sun(Time(T0, format="unix"))  # geocentric apparent direction
        self.assertLess(float(separation_deg(ra, dec, g.ra.deg, g.dec.deg)), 1.0)
        self.assertLess(float(separation_deg(ra, dec, *_sun_low_precision(T0))), 1.0)
        # cached for up to 60 s
        self.assertIs(self.ks.body_positions(T0 + 30)["Sun"], self.ks.body_positions(T0)["Sun"])
        lab = self.ks.classify(ra + 0.5, dec, 45.0, T0 + 10)
        self.assertEqual((lab["kind"], lab["name"]), ("solar", "Sun"))

    def test_set_sun_labels_nothing(self):
        """
        Low in the west after sunset the body test's radius, widened towards the
        horizon, reaches the set Sun; a pixel there must not be called solar,
        while the same separation with the Sun up still is.
        """
        import astropy.units as u
        from astropy.coordinates import AltAz, EarthLocation, SkyCoord
        from astropy.time import Time

        loc = EarthLocation(lat=34.348361 * u.deg, lon=-106.885778 * u.deg, height=1477 * u.m)
        # the first time after T0 with the Sun 6-10 deg below the horizon
        t = next(T0 + 600.0 * k for k in range(200)
                 if -10 < altaz(*self.ks.body_positions(T0 + 600.0 * k)["Sun"], T0 + 600.0 * k)[0] < -6)
        sra, sdec = self.ks.body_positions(t)["Sun"]
        _, saz = altaz(sra, sdec, t)
        frame = AltAz(obstime=Time(t, format="unix"), location=loc)
        c = SkyCoord(alt=12.0 * u.deg, az=float(saz) * u.deg, frame=frame).icrs
        sep = float(separation_deg(c.ra.deg, c.dec.deg, sra, sdec))
        self.assertLess(sep, self.ks.body_beams * elevation_radius(self.ks.beam, 12.0))
        self.assertFalse(self.ks.body_up("Sun", t))
        lab = self.ks.classify(c.ra.deg, c.dec.deg, 12.0, t)
        self.assertTrue(lab is None or lab["kind"] != "solar")

    def test_pulsar_needs_position_and_dm(self):
        p = next(p for p in self.ks.pulsars if p["name"] == "B0950+08")
        lab = self.ks.classify(p["ra_deg"] + 0.3, p["dec_deg"], 60.0, T0, dm=3.1, dm_step=0.6)
        self.assertEqual((lab["kind"], lab["name"]), ("pulsar", "B0950+08"))
        self.assertTrue(lab["dm_match"])
        far = self.ks.classify(p["ra_deg"], p["dec_deg"], 60.0, T0, dm=40.0, dm_step=0.6)
        self.assertTrue(far is None or far["kind"] != "pulsar", far)

    def test_crab_giant_pulse_is_told_from_tau_a_by_its_dm(self):
        """The Crab pulsar is in Tau A: a pulse at its DM is the pulsar, a trigger at DM ~1 the nebula"""
        crab = next(p for p in self.ks.pulsars if p["name"] == "B0531+21")
        self.assertEqual(crab["alias"], "Crab")
        self.assertTrue(crab["giant_pulses"])
        lab = self.ks.classify(83.633, 22.014, 70.0, T0, dm=57.4, dm_step=0.6)
        self.assertEqual((lab["kind"], lab["name"], lab["alias"]), ("pulsar", "B0531+21", "Crab"))
        self.assertTrue(lab["giant_pulse"])
        lab = self.ks.classify(83.633, 22.014, 70.0, T0, dm=1.2, dm_step=0.6)
        self.assertEqual((lab["kind"], lab["name"]), ("bright_source", "Tau A"))

    def test_bright_source(self):
        lab = self.ks.classify(299.868, 40.734 + 0.5, 60.0, T0, dm=300.0, dm_step=1.0)
        self.assertEqual((lab["kind"], lab["name"]), ("bright_source", "Cyg A"))
        self.assertIsNone(self.ks.classify(299.868, 40.734 + 25.0, 60.0, T0, dm=300.0))

    def test_csv_extends_and_overrides(self):
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "psr.csv")
            with open(p, "w") as f:
                f.write("name,ra_deg,dec_deg,dm,period_s\nB0950+08,148.2888,7.9266,3.0,0.253\n"
                        "J9999+00,10.0,0.0,100.0,\n")
            psrs = {q["name"]: q for q in load_pulsars(p)}
            self.assertEqual(psrs["B0950+08"]["dm"], 3.0)
            self.assertIsNone(psrs["J9999+00"]["period_s"])
            self.assertEqual(len(psrs), len(load_pulsars()) + 1)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestNarrowband(unittest.TestCase):
    def _quality(self, dyn, row, width=2):
        D = BAND.nchan
        edges = BAND.edges_mhz()
        dmt = fdmt(dyn[:, :, None].astype(np.float32), fdmt_plan(edges, D))[:, :, 0]
        box = np.convolve(dmt[row], np.ones(width), "full")[:dmt.shape[1]]
        t = int(np.argmax(box))
        widths = [1, 2, 4, 8]
        return candidate_quality(dyn, dmt, np.ones((D + 1, len(widths))), edges, row, t, width,
                                 widths)

    def test_dispersed_burst_is_broadband(self):
        rng = np.random.default_rng(1)
        dt = 0.0836
        edges = BAND.edges_mhz()
        delay = 30
        dm = delay * dt / (K_DM * (edges[0] ** -2 - edges[-1] ** -2))
        dyn = dispersed(edges, dt, 160, dm, t_top=50 * dt, amp=20.0)
        dyn = dyn + rng.standard_normal(dyn.shape)
        q = self._quality(dyn, delay)
        self.assertGreater(q["n_eff_channels"], 40, q)
        self.assertLess(q["top2_fraction"], 0.2, q)

    def test_single_channel_spike_is_narrowband(self):
        rng = np.random.default_rng(2)
        dyn = rng.standard_normal((BAND.nchan, 160))
        dyn[40, 80:82] += 100.0
        q = self._quality(dyn, 20)
        self.assertLess(q["n_eff_channels"], 3, q)
        self.assertGreater(q["top2_fraction"], 0.6, q)


class CarrierSimulator:
    """
    A SkySimulator whose images also carry a steady carrier in a few channels
    at one position (an aircraft or meteor-scatter signal, spread over
    neighbouring channels by its Doppler shift) for a few seconds. amp_sigma
    is in units of the simulator's image noise; after regridding and
    normalisation 19 gives about 11 sigma, below the normaliser's clip of 12
    (a stronger sample is set to zero; a carrier in a single channel between
    the clip and the threshold is barely detectable). The carrier is
    therefore found only once the averaging tiers (2 and up) have started,
    about 90 s in.
    """

    def __init__(self, sim, ra, dec, chan, t_start, duration_s, amp_sigma=19.0):
        self.sim, self.ra, self.dec, self.chan = sim, ra, dec, chan
        self.t0, self.t1, self.amp = t_start, t_start + duration_s, amp_sigma

    def next_image(self):
        from skymon.burst import stamp

        sim = self.sim
        t = sim._t(sim.n)
        meta, img = sim.next_image()
        if self.t0 <= t < self.t1:
            x, y = sim.wcs.all_world2pix(self.ra, self.dec, 1, ra_dec_order=True)
            G = sim.grid
            ix = int(np.clip(round(float(x)) - 1, 0, G - 1))
            iy = int(np.clip(round(float(y)) - 1, 0, G - 1))
            frac = np.zeros(len(sim.centres))
            frac[np.atleast_1d(self.chan)] = 1.0
            amp = self.amp * sim.sigma[:, iy, ix].astype(np.float64)
            stamp(img, None, float(x), float(y), amp, frac, sim.centres, sim.grid_res)
        return meta, img


def _high_position(unix, avoid, min_sep=25.0, el_min=65.0):
    """A position above el_min elevation, at least min_sep from every (ra, dec) in avoid"""
    for dec in (34.0, 45.0, 25.0, 55.0, 15.0):
        for r in range(0, 360, 3):
            if altaz(r, dec, unix)[0] < el_min:
                continue
            if all(separation_deg(r, dec, a, d) > min_sep for a, d in avoid):
                return float(r), dec
    raise AssertionError("no position found")


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from skymon.monitor import Config, Monitor
        from skymon.simulate import SkySimulator, tier_dt_function

        cls.out = tempfile.mkdtemp()
        band = Band(2336, 64)
        f_top = band.f_hi
        psr = next(p for p in load_pulsars() if p["name"] == "B1508+55")
        assert altaz(psr["ra_deg"], psr["dec_deg"], T0)[0] > 60
        cls.psr = psr
        burst = dict(id="sim-psr", ra=psr["ra_deg"], dec=psr["dec_deg"], dm=psr["dm"], snr=30.0,
                     width_s=0.005, t_inf=T0 + 45.0 - K_DM * psr["dm"] / f_top ** 2)
        sim = SkySimulator(chan0=2336, nchan=64, start_unix=T0, frb_per_hour=0, rfi_per_hour=0,
                           rfi_channel=False, seed=7, frb_list=[burst], el_min=25.0,
                           tier_dt=tier_dt_function(band.edges_mhz(), 0.083592, 64, 300.0, 64))
        ks = KnownSources(beam_fwhm_deg(band.f_centre))
        avoid = [(p["ra_deg"], p["dec_deg"]) for p in ks.pulsars] + \
            list(ks.body_positions(T0).values()) + [(299.868, 40.734), (350.850, 58.815),
                                                    (83.633, 22.014), (187.706, 12.391),
                                                    (252.784, 4.993), (139.524, -12.096)]
        cls.carrier = _high_position(T0, avoid)
        # a solar burst: the "Sun" is put at a high position clear of the
        # others, and a burst with no dispersion goes off there
        cls.sun = _high_position(T0, avoid + [cls.carrier])
        sim.frb_list.append(dict(id="sim-sun", ra=cls.sun[0], dec=cls.sun[1], dm=0.0, snr=400.0,
                                 width_s=0.2, t_inf=T0 + 130.0))
        # a faint dispersed burst elsewhere during it, 40 times fainter: what a
        # sidelobe of the solar burst looks like. It reaches the top of the band
        # as the solar burst does
        cls.side = _high_position(T0, avoid + [cls.carrier, cls.sun], el_min=45.0)
        sim.frb_list.append(dict(id="sim-side", ra=cls.side[0], dec=cls.side[1], dm=20.0, snr=10.0,
                                 width_s=0.2, t_inf=T0 + 130.0 - K_DM * 20.0 / f_top ** 2))
        sim.frb_list.sort(key=lambda f: f["t_inf"] + K_DM * f["dm"] / f_top ** 2)
        src = CarrierSimulator(sim, *cls.carrier, chan=[40, 41, 42], t_start=T0 + 100.0, duration_s=8.0)
        cls.mon = Monitor(Config(out_dir=cls.out, nside=32, dm_max=300.0, block0=64, nthreads=4,
                                 warmup_images=32, inject_interval_s=0, seed=7, accum_hours=0.5))
        cls._body_positions = KnownSources.body_positions
        KnownSources.body_positions = lambda self, unix: {"Sun": cls.sun, "Jupiter": (0.0, -89.0)}
        for _ in range(int(170 / sim.dt)):
            meta, img = src.next_image()
            cls.mon.process(meta, img)
        cls.mon.store.flush()
        cls.cands = [c.to_dict() for c in cls.mon.store.list()]
        cond = cls.mon.cond
        cls.n_solar = cond.n_solar + sum(x.get("solar", 0) for x in cond.series)

    @classmethod
    def tearDownClass(cls):
        KnownSources.body_positions = cls._body_positions
        shutil.rmtree(cls.out, ignore_errors=True)

    def _near(self, ra, dec, radius):
        return [c for c in self.cands if separation_deg(ra, dec, c["ra"], c["dec"]) < radius]

    def test_pulsar_pulse_is_tallied_not_stored(self):
        """A single pulse of an ordinary pulsar is counted for its pulsar, not stored, and opens no event"""
        p = self.psr
        near = self._near(p["ra_deg"], p["dec_deg"], 3 * self.mon.beam)
        self.assertFalse([c for c in near if c["cls"] in ("known", "candidate")],
                         [(c["cls"], c["dm"], c["rfi_reasons"]) for c in near])
        tally = self.mon.pulsar_tally()
        self.assertIn("B1508+55", tally, tally)
        t, snr, dm = tally["B1508+55"][0]
        self.assertLess(abs(dm - p["dm"]), 0.1 * p["dm"] + 2)
        self.assertGreater(snr, 8)

    def test_carrier_is_narrowband_rfi_with_data(self):
        near = self._near(*self.carrier, 3 * self.mon.beam)
        nb = [c for c in near if c["cls"] == "rfi" and
              "narrowband (a few channels)" in c["rfi_reasons"]]
        self.assertTrue(nb, [(c["cls"], c["dm"], c["rfi_reasons"], c["quality"].get(
            "n_eff_channels")) for c in near])
        for c in nb:
            self.assertFalse(c["rfi_certain"])
            self.assertTrue(os.path.exists(c["files"]["npz"]))
        self.assertFalse([c for c in near if c["cls"] in ("candidate", "known")],
                         [(c["cls"], c["quality"]) for c in near])

    def test_solar_burst_is_labelled_not_rfi(self):
        near = self._near(*self.sun, 3 * self.mon.beam)
        solar = [c for c in near if c["cls"] == "known" and
                 (c.get("known_source") or {}).get("kind") == "solar"]
        self.assertTrue(solar, [(c["cls"], c["dm"], c["rfi_reasons"]) for c in near])
        best = max(solar, key=lambda c: c["snr"])
        self.assertTrue(best["known_source"]["burst"])
        self.assertIn("DM near zero", best["quality"]["rfi_tests"])  # kept as a note
        self.assertFalse(best["rfi_reasons"])
        self.assertTrue(os.path.exists(best["files"]["npz"]))  # its data are kept
        self.assertFalse([c for c in near if c["cls"] in ("rfi", "candidate")],
                         [(c["cls"], c["dm"], c["rfi_reasons"]) for c in near])
        self.assertGreaterEqual(self.n_solar, 1)  # counted apart from RFI

    def test_faint_burst_during_a_bright_one_is_its_sidelobe(self):
        near = self._near(*self.side, 3 * self.mon.beam)
        self.assertTrue(near, "the faint burst was not found at all")
        side = [c for c in near if any(r.startswith("sidelobe of a solar burst")
                                       for r in c["rfi_reasons"])]
        self.assertTrue(side, [(c["cls"], c["snr"], c["dm"], c["rfi_reasons"]) for c in near])
        for c in side:
            self.assertEqual(c["cls"], "rfi")
            self.assertFalse(c["rfi_certain"])  # its data are kept
            self.assertTrue(os.path.exists(c["files"]["npz"]))
        self.assertFalse([c for c in near if c["cls"] == "candidate"],
                         [(c["cls"], c["snr"], c["dm"], c["rfi_reasons"]) for c in near])


if __name__ == "__main__":
    unittest.main()
