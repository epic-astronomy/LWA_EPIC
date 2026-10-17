"""
Tests of the external context attached to the combiner's events
(src/python/skymon/context.py). No test uses the network: the SWPC payloads
are small real samples saved in data/ (NOAA data, public domain), and the
unreachable source is a closed local port.

    cd tests/python && python -m unittest -v test_context
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src", "python"))

from skymon import context as cx  # noqa: E402
from skymon.dedisp import K_DM  # noqa: E402
from skymon.geometry import LWASV_LAT, LWASV_LON  # noqa: E402

from test_combiner import BANDS, T0, FakeMonitor, make, zenith_ish  # noqa: E402

DATA = os.path.join(HERE, "data")
UNREACHABLE = "http://127.0.0.1:9/none.json"  # the discard port: refused at once


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def event(t_bottom_lo, dm=300.0, bands=BANDS, ra=180.0, dec=0.0):
    """An event whose arrival at the bottom of the lowest band is t_bottom_lo"""
    f_low = min(b[0] for b in bands.values())
    t_inf = t_bottom_lo - K_DM * dm / f_low ** 2
    return dict(id="ev-test", t_inf=t_inf, dm=dm, ra=ra, dec=dec,
                bands={n: dict(f_lo=b[0], f_hi=b[1]) for n, b in bands.items()})


class TestParsing(unittest.TestCase):
    def test_edited_events_sample(self):
        recs = cx.parse_edited_events(load("swpc_edited_events_sample.json"))
        # the 245 MHz fixed-frequency burst (RBR) is not kept
        self.assertEqual(sorted({r["type"] for r in recs}), ["RSP", "XRA"])
        self.assertEqual(len(recs), 6)
        ii = next(r for r in recs if r["type"] == "RSP" and r["observatory"] == "SAG")
        self.assertEqual(ii["sweep"], "II")
        self.assertEqual(ii["cls"], "II/1")
        self.assertEqual(ii["f_mhz"], [25.0, 51.0])
        self.assertEqual(ii["speed_km_s"], 929.0)
        self.assertEqual(cx.iso(ii["begin"]), "2026-10-10T15:15:12Z")  # UTC without a zone
        self.assertEqual(cx.iso(ii["end"]), "2026-10-10T15:18:30Z")
        self.assertIsNone(ii["max"])
        xra = [r for r in recs if r["type"] == "XRA"]
        self.assertEqual({r["cls"] for r in xra}, {"B9.5", "B9.1"})
        self.assertEqual(cx.iso(xra[0]["max"]), "2026-10-17T05:23:00Z")

    def test_goes_flare_samples(self):
        week = cx.parse_goes_flares(load("swpc_goes_xray_flares_7day_sample.json"))
        self.assertEqual([r["cls"] for r in week], ["C1.7", "C2.7", "M1.2"])
        self.assertEqual(cx.iso(week[2]["begin"]), "2026-10-10T13:56:00Z")
        self.assertEqual(week[0]["observatory"], "G18")
        latest = cx.parse_goes_flares(load("swpc_goes_xray_flares_latest_sample.json"))
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0]["cls"], "C1.2")

    def test_malformed_records_are_skipped(self):
        recs = cx.parse_edited_events([None, 3, {"type": "RSP"}, {"type": "RSP",
                                       "begin_datetime": "garbage"},
                                       {"type": "RSP", "begin_datetime": "2026-10-10T15:00:00",
                                        "frequency": "??"}])
        self.assertEqual(len(recs), 1)
        self.assertIsNone(recs[0]["f_mhz"])
        self.assertEqual(cx.parse_edited_events({"not": "a list"}), [])
        self.assertEqual(cx.parse_goes_flares([{"begin_time": "Unk"}]), [])
        ongoing = cx.parse_goes_flares([{"begin_time": "2026-10-10T15:00:00Z",
                                         "end_time": "Unk", "max_class": "C3.0"}])
        self.assertTrue(ongoing[0]["ongoing"])


class TestMatching(unittest.TestCase):
    def setUp(self):
        self.recs = cx.parse_edited_events(load("swpc_edited_events_sample.json")) + \
            cx.parse_goes_flares(load("swpc_goes_xray_flares_7day_sample.json"))

    def test_sweeps_match_by_time_and_frequency(self):
        ev = event(cx.parse_time("2026-10-10T15:25:00"))
        got = cx.match_solar(self.recs, ev, now=T0)
        sweeps = {(m["cls"], m["observatory"]): m["bands"] for m in got if m["type"] == "RSP"}
        # 25-51 MHz reaches only the low band; 41-156 MHz reaches mid and hi
        self.assertEqual(sweeps[("II/1", "SAG")], ["lo"])
        self.assertEqual(sweeps[("II/2", "SVI")], ["hi", "mid"])
        # 113-180 MHz is above every band
        self.assertNotIn(("III/2", "SVI"), sweeps)
        # the M1.2 flare ended at 14:27, more than 15 min before the earliest arrival
        self.assertFalse([m for m in got if m["type"] == "XRA"])

    def test_time_window(self):
        far = event(cx.parse_time("2026-10-10T16:30:00"), dm=10.0)
        self.assertEqual(cx.match_solar(self.recs, far, now=T0), [])
        # a type III at 20:40:24-20:40:42 down to 25 MHz: 14 min later still matches
        near = event(cx.parse_time("2026-10-10T20:54:30"), dm=10.0)
        got = cx.match_solar(self.recs, near, now=T0)
        self.assertEqual([m["cls"] for m in got], ["III/1"])
        self.assertEqual(got[0]["begin"], "2026-10-10T20:40:24Z")

    def test_sweep_to_the_spectrograph_floor_reaches_lower_bands(self):
        rec = dict(type="RSP", cls="III/2", begin=T0, end=T0 + 30, f_mhz=[25.0, 80.0],
                   observatory="X", source="swpc")
        ev = event(T0 + 60, dm=5.0, bands={"vlo": (16.0, 19.0)})
        self.assertEqual(len(cx.match_solar([rec], ev, now=T0)), 1)
        rec["f_mhz"] = [35.0, 80.0]
        self.assertEqual(cx.match_solar([rec], ev, now=T0), [])

    def test_flare_reported_twice_is_listed_once(self):
        ev = event(cx.parse_time("2026-10-17T05:25:00"), dm=5.0)
        got = cx.match_solar(self.recs, ev, now=T0)
        self.assertEqual(len([m for m in got if m["type"] == "XRA"]), 1)

    def test_sun_elevation(self):
        day, _ = cx.sun_position(cx.parse_time("2026-10-10T19:00:00"))  # local noon
        night, _ = cx.sun_position(cx.parse_time("2026-10-11T07:00:00"))
        self.assertGreater(day, 40.0)
        self.assertLess(night, -40.0)
        _, sep = cx.sun_position(T0, 0.0, 90.0)
        self.assertTrue(80 < sep < 110)  # the pole is ~90 deg from the Sun


class TestLightning(unittest.TestCase):
    def test_counting_by_distance_and_time(self):
        self.assertAlmostEqual(cx.distance_km(0, 0, 0, 1), 111.19, places=1)
        ev = event(T0 + 100, dm=10.0)
        dlat = 1.0 / 111.19  # one km of latitude
        strikes = cx.parse_strikes({"strikes": [
            dict(time=T0, lat=LWASV_LAT + 20 * dlat, lon=LWASV_LON),       # 20 km
            dict(time=T0 + 30, lat=LWASV_LAT - 90 * dlat, lon=LWASV_LON),  # 90 km
            dict(t=T0 + 60, latitude=LWASV_LAT + 150 * dlat, longitude=LWASV_LON),  # 150 km
            dict(time=T0 - 3600, lat=LWASV_LAT, lon=LWASV_LON),            # an hour earlier
            [cx.iso(T0 + 5), LWASV_LAT + 5 * dlat, LWASV_LON],             # 5 km, as a list
            dict(time="bad", lat=0, lon=0), dict(lat=1)]})
        self.assertEqual(len(strikes), 5)
        got = cx.match_lightning(strikes, ev, covered_until=T0 + 7200)
        self.assertEqual(got["strikes_100km"], 3)
        self.assertAlmostEqual(got["nearest_km"], 5.0, delta=0.2)
        self.assertTrue(got["complete"])
        self.assertFalse(cx.match_lightning(strikes, ev, covered_until=T0)["complete"])
        none = cx.match_lightning([], ev, covered_until=T0 + 7200)
        self.assertEqual((none["strikes_100km"], none["nearest_km"]), (0, None))

    def test_file_source_keeps_nearby_strikes_across_fetches(self):
        import time
        d = tempfile.mkdtemp()
        try:
            now = time.time()
            path = os.path.join(d, "strikes.json")
            with open(path, "w") as f:
                json.dump([dict(time=now - 60, lat=LWASV_LAT, lon=LWASV_LON + 0.5),
                           dict(time=now - 60, lat=0.0, lon=0.0)], f)  # far away: dropped
            prov = cx.LightningProvider(path)
            data = prov.fetch(None)
            self.assertEqual(len(data["strikes"]), 1)
            with open(path, "w") as f:
                json.dump([dict(time=now - 30, lat=LWASV_LAT, lon=LWASV_LON)], f)
            self.assertEqual(len(prov.fetch(data)["strikes"]), 2)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestService(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_unreachable_provider_is_marked_unavailable(self):
        svc = cx.ContextService([cx.SwpcProvider(urls=dict(edited_events=UNREACHABLE),
                                                 timeout_s=2.0),
                                 cx.LightningProvider(UNREACHABLE, timeout_s=2.0)], self.dir)
        self.assertFalse(svc.refresh("swpc"))
        self.assertFalse(svc.refresh("lightning"))
        self.assertIsNotNone(svc.status()["swpc"]["error"])
        ctx = svc.for_event(event(T0))
        self.assertIsNone(ctx["solar"])
        self.assertIsNone(ctx["lightning"])
        self.assertEqual(ctx["sources"], [])
        self.assertEqual(sorted(ctx["unavailable"]), ["lightning", "swpc"])
        self.assertIsNotNone(ctx["sun_el_deg"])  # computed locally

    def test_last_good_data_survive_failures_and_restarts(self):
        prov = cx.SwpcProvider(urls=dict(edited_events="file://" + os.path.join(
            DATA, "swpc_edited_events_sample.json")))
        svc = cx.ContextService([prov], self.dir)
        self.assertTrue(svc.refresh("swpc"))
        prov.urls = dict(edited_events=UNREACHABLE)
        prov.timeout_s = 2.0
        self.assertFalse(svc.refresh("swpc"))
        ev = event(cx.parse_time("2026-10-10T15:25:00"))
        self.assertEqual(len(svc.for_event(ev)["solar"]), 2)
        self.assertEqual(svc.for_event(ev)["sources"], ["swpc"])
        # a new service (a restart without network) starts from the disk cache
        svc2 = cx.ContextService([cx.SwpcProvider(urls=dict(edited_events=UNREACHABLE))],
                                 self.dir)
        self.assertEqual(len(svc2.for_event(ev)["solar"]), 2)

    def test_background_threads_stop(self):
        svc = cx.ContextService([cx.LightningProvider(UNREACHABLE, interval_s=0.05,
                                                      timeout_s=1.0)], self.dir)
        stop = threading.Event()
        svc.start(stop)
        stop.wait(0.3)
        stop.set()
        for t in svc.threads:
            t.join(5)
            self.assertFalse(t.is_alive())
        self.assertGreaterEqual(svc.status()["lightning"]["attempts"], 2)


class TestCombinerContext(unittest.TestCase):
    def setUp(self):
        self.ra, self.dec = zenith_ish(T0)

    def run_until_decided(self, comb, t, step=10.0, n=2000):
        for _ in range(n):
            comb.poll(now=t)
            if all(e["status"] != "pending" for e in comb.events.values()):
                return t
            t += step
        return t

    def test_context_attached_to_a_decided_event(self):
        dm = 300.0
        nodes = {n: FakeMonitor(n, b, followup_snr=9.0) for n, b in BANDS.items()}
        nodes["hi"].candidate("c1", 15.0, dm, T0, self.ra, self.dec)
        comb, out = make(nodes, lightning_url=UNREACHABLE)
        try:
            # a type III sweep over 25-80 MHz at the time of the high band's arrival
            t_hi = T0 + K_DM * dm / BANDS["hi"][0] ** 2
            sweep = cx.parse_edited_events([dict(
                type="RSP", begin_datetime=cx.iso(t_hi - 120)[:-1],
                end_datetime=cx.iso(t_hi - 60)[:-1], frequency="025-080",
                particulars1="III/3", observatory="SAG")])
            comb.context.set_data("swpc", dict(edited_events=sweep))
            dlat = 1.0 / 111.19
            comb.context.set_data("lightning", dict(
                strikes=[(t_hi, LWASV_LAT + 40 * dlat, LWASV_LON)], until=T0 + 86400))
            self.run_until_decided(comb, T0 + 300)
            (ev,) = comb.events.values()
            self.assertEqual(ev["status"], "confirmed")
            ctx = ev["context"]
            self.assertEqual(sorted(ctx["sources"]), ["lightning", "swpc"])
            self.assertEqual(len(ctx["solar"]), 1)
            self.assertEqual(ctx["solar"][0]["cls"], "III/3")
            self.assertEqual(ctx["solar"][0]["bands"], ["hi", "lo", "mid"])
            self.assertEqual(ctx["lightning"]["strikes_100km"], 1)
            self.assertAlmostEqual(ctx["lightning"]["nearest_km"], 40.0, delta=0.5)
            self.assertTrue(-90 <= ctx["sun_el_deg"] <= 90)
            # the context is saved with the event
            files = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(out, "events"))
                     for f in fs if f.endswith(".json")]
            with open(files[0]) as f:
                self.assertEqual(json.load(f)["context"]["solar"][0]["cls"], "III/3")
            # a report arriving after the decision updates the context
            comb.context.set_data("swpc", dict(edited_events=sweep + cx.parse_edited_events(
                [dict(type="XRA", begin_datetime=cx.iso(t_hi)[:-1], particulars1="M2.0")])))
            comb.poll(now=T0 + 3000)
            self.assertEqual(len(ev["context"]["solar"]), 2)
            # a late detection reopens the event, which is given its context again
            nodes["lo"].candidate("c2", 12.0, dm, T0, self.ra, self.dec)
            comb.poll(now=T0 + 3010)
            self.assertIn("context", ev)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_combiner_runs_with_unreachable_sources(self):
        nodes = {n: FakeMonitor(n, b, followup_snr=9.0) for n, b in BANDS.items()}
        nodes["hi"].candidate("c1", 15.0, 200.0, T0, self.ra, self.dec)
        comb, out = make(nodes, lightning_url=UNREACHABLE)
        comb.context.providers["swpc"].urls = dict(edited_events=UNREACHABLE)
        for p in comb.context.providers.values():
            p.timeout_s = 2.0
        stop = threading.Event()
        try:
            comb.context.start(stop)
            self.run_until_decided(comb, T0 + 300)
            (ev,) = comb.events.values()
            self.assertEqual(ev["status"], "confirmed")
            self.assertEqual(ev["context"]["sources"], [])
            self.assertEqual(sorted(ev["context"]["unavailable"]), ["lightning", "swpc"])
            self.assertIn("context", comb.status())
        finally:
            stop.set()
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
