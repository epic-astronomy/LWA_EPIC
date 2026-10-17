"""
Tests of the combiner (src/python/skymon/combiner.py): the consensus of the
monitors of several bands.

    cd tests/python && python -m unittest -v test_combiner
"""
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src", "python"))

from skymon.combiner import Combiner, CombinerConfig, fit_dm  # noqa: E402
from skymon.dedisp import K_DM  # noqa: E402
from skymon.geometry import altaz  # noqa: E402

T0 = 1792008000.0
BANDS = {"hi": (74.27, 77.33), "mid": (55.88, 58.94), "lo": (32.97, 36.03)}


def zenith_ish(unix):
    """A position high in the sky at unix"""
    for ra in np.arange(0, 360, 5.0):
        if altaz(ra, 34.0, unix)[0] > 80:
            return float(ra), 34.0
    raise RuntimeError


class FakeMonitor:
    """Answers the combiner like a monitor would, from scripted state"""

    def __init__(self, name, band, up=True, followup_snr=None):
        self.name, self.band, self.up = name, band, up
        self.cands, self.watches = [], {}
        self.followup_snr = followup_snr

    def status(self):
        f_lo, f_hi = self.band
        return dict(band=dict(f_lo=f_lo, f_hi=f_hi), el_min=10.0, beam_deg=3.0, threshold=8.0,
                    tiers=[dict(dm_max=3000.0, latency_s=60.0)])

    def candidate(self, cid, snr, dm, t_inf, ra, dec, cls="candidate", injection_id=None,
                  true_dm=None):
        """true_dm: the burst's DM, which sets the measured arrival; dm is the reported one"""
        f_lo, f_hi = self.band
        t_bottom = t_inf + K_DM * (dm if true_dm is None else true_dm) / f_lo ** 2
        self.cands.append(dict(id=cid, cls=cls, snr=snr, dm=dm, dm_step=1.0,
                               t_inf=t_bottom - K_DM * dm / f_lo ** 2,
                               t_bottom=t_bottom, ra=ra, dec=dec, el=80.0,
                               tier=1, tier_dt=0.167, width_s=0.167, f_lo_mhz=f_lo,
                               f_hi_mhz=f_hi, injection_id=injection_id))

    def get(self, path):
        if not self.up:
            raise ConnectionError
        if path.startswith("/api/status"):
            return self.status()
        if path.startswith("/api/candidates"):
            return list(self.cands)
        if path.startswith("/api/watches"):
            out = []
            for w in self.watches.values():
                w = dict(w)
                if self.followup_snr is not None:
                    w["state"] = "measured"
                    w["result"] = dict(snr=self.followup_snr, dm=w["dm"], t_inf=w["t_inf"])
                out.append(w)
            return out
        raise KeyError(path)

    def post(self, path, body):
        if not self.up:
            raise ConnectionError
        if path == "/api/watch":
            self.watches[body["id"]] = dict(body, state="pending")
            return body
        if path == "/api/inject":
            self.injected = body
            return dict(accepted=True)
        raise KeyError(path)


def make(nodes, **kw):
    out = tempfile.mkdtemp()
    comb = Combiner(CombinerConfig(nodes={n: f"http://{n}" for n in nodes}, out_dir=out,
                                   settle_s=0.0, **kw))
    comb._get = lambda node, path: nodes[node["name"]].get(path)
    comb._post = lambda node, path, body: nodes[node["name"]].post(path, body)
    return comb, out


class TestFit(unittest.TestCase):
    def test_dm_and_time_from_three_bands(self):
        dm, t_inf = 412.3, T0
        dets = [(f, t_inf + K_DM * dm / f ** 2, 0.1) for f in (32.97, 55.88, 74.27)]
        got = fit_dm(dets)
        self.assertAlmostEqual(got[0], dm, places=3)
        self.assertAlmostEqual(got[1], t_inf, places=2)
        self.assertIsNone(fit_dm(dets[:1]))


class TestConsensus(unittest.TestCase):
    def setUp(self):
        self.ra, self.dec = zenith_ish(T0)

    def run_until_decided(self, comb, start, step=10.0, n=2000):
        t = start
        for _ in range(n):
            comb.poll(now=t)
            if all(e["status"] != "pending" for e in comb.events.values()):
                return t
            t += step
        return t

    def test_confirmed_by_followup_in_other_bands(self):
        dm = 300.0
        nodes = {n: FakeMonitor(n, b, followup_snr=9.0) for n, b in BANDS.items()}
        nodes["hi"].candidate("c1", 15.0, dm, T0, self.ra, self.dec)
        comb, out = make(nodes)
        try:
            self.run_until_decided(comb, T0 + 300)
            (ev,) = comb.events.values()
            self.assertEqual(ev["status"], "confirmed")
            self.assertEqual(set(ev["seen"]), {"hi", "mid", "lo"})
            # the watches predicted the arrival in each band from the dispersion law
            w = next(iter(nodes["lo"].watches.values()))
            self.assertAlmostEqual(w["t_inf"], T0, places=3)
            self.assertGreater(w["t_tol"], 0)
            self.assertTrue(os.listdir(os.path.join(out, "events")))
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_two_detections_refit_the_dm(self):
        dm = 250.0
        nodes = {n: FakeMonitor(n, b, followup_snr=2.0) for n, b in BANDS.items()}
        # each band reports a DM about one trial step off, but measures the arrival well
        nodes["hi"].candidate("a", 12.0, dm + 0.8, T0, self.ra, self.dec, true_dm=dm)
        nodes["mid"].candidate("b", 11.0, dm - 0.9, T0, self.ra + 0.5, self.dec, true_dm=dm)
        comb, out = make(nodes)
        try:
            self.run_until_decided(comb, T0 + 400)
            (ev,) = comb.events.values()
            self.assertEqual(ev["status"], "confirmed")
            self.assertGreaterEqual(ev["refits"], 1)
            self.assertLess(abs(ev["dm"] - dm), 0.1)  # the fit beats either band alone
            self.assertEqual(ev["bands"]["lo"]["state"], "followup")
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_wrong_dm_detection_of_the_same_burst_joins_its_event(self):
        """A bright burst also triggers a neighbouring trial DM during its sweep: one event"""
        dm = 280.0
        nodes = {n: FakeMonitor(n, b, followup_snr=9.0) for n, b in BANDS.items()}
        nodes["hi"].candidate("true", 25.0, dm, T0, self.ra, self.dec)
        # the same burst found at DM 372, arriving at the band bottom within the sweep
        f_lo, f_hi = BANDS["hi"]
        t_mid = T0 + K_DM * dm * 0.5 * (f_lo ** -2 + f_hi ** -2)
        nodes["hi"].candidate("side", 14.0, 372.0, t_mid - K_DM * 372.0 / f_lo ** 2,
                              self.ra + 0.4, self.dec)
        comb, out = make(nodes)
        try:
            self.run_until_decided(comb, T0 + 400)
            self.assertEqual(len(comb.events), 1)
            (ev,) = comb.events.values()
            self.assertEqual(ev["bands"]["hi"]["candidate"]["id"], "true")
            self.assertEqual(ev["bands"]["hi"]["others"], ["side"])
            self.assertLess(abs(ev["dm"] - dm), 1.0)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_one_band_with_a_plausible_spectrum_is_band_limited(self):
        """Seen at 57 MHz only: emission confined to a few MHz explains the other bands"""
        nodes = {n: FakeMonitor(n, b, followup_snr=0.5) for n, b in BANDS.items()}
        nodes["mid"].candidate("c1", 12.0, 120.0, T0, self.ra, self.dec)
        comb, out = make(nodes)
        try:
            self.run_until_decided(comb, T0 + 200)
            (ev,) = comb.events.values()
            self.assertEqual(ev["status"], "band_limited")
            self.assertEqual(ev["seen"], ["mid"])
            self.assertTrue(ev["spectrum"]["plausible"])
            self.assertTrue(any("confined" in n for n in ev["spectrum"]["consistent_with"]))
            self.assertEqual(ev["review"], "pending")
            self.assertEqual(ev["best_snr"], 12.0)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_bright_burst_missing_from_the_next_band_is_still_band_limited(self):
        """
        Nothing 4.6 MHz below a bright detection at the top band is still a burst
        whose emission peaks just above it: kept as band_limited
        """
        bands = dict(BANDS, hi2=(69.67, 72.73))
        nodes = {n: FakeMonitor(n, b, followup_snr=0.0) for n, b in bands.items()}
        nodes["hi"].candidate("c1", 20.0, 120.0, T0, self.ra, self.dec)
        comb, out = make(nodes)
        try:
            self.run_until_decided(comb, T0 + 200)
            (ev,) = comb.events.values()
            self.assertEqual(ev["status"], "band_limited")
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_contradiction_between_overlapping_bands_is_implausible(self):
        """A bright detection and nothing in a band covering nearly the same frequencies"""
        bands = dict(BANDS, hi2=(73.77, 76.83))
        nodes = {n: FakeMonitor(n, b, followup_snr=0.0) for n, b in bands.items()}
        nodes = {n: FakeMonitor(n, b, followup_snr=0.0) for n, b in bands.items()}
        nodes["hi"].candidate("c1", 20.0, 120.0, T0, self.ra, self.dec)
        comb, out = make(nodes)
        try:
            self.run_until_decided(comb, T0 + 200)
            (ev,) = comb.events.values()
            self.assertEqual(ev["status"], "implausible")
            self.assertFalse(ev["spectrum"]["plausible"])
            self.assertEqual(ev["review_priority"], 4)  # kept, reviewed last
            self.assertTrue(os.listdir(os.path.join(out, "events")))
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_review_verdicts_are_kept(self):
        from skymon.combiner import serve as serve_combiner
        import urllib.request
        nodes = {n: FakeMonitor(n, b, followup_snr=9.0) for n, b in BANDS.items()}
        nodes["hi"].candidate("c1", 15.0, 200.0, T0, self.ra, self.dec)
        comb, out = make(nodes)
        httpd = serve_combiner(comb, host="127.0.0.1", port=0)
        try:
            self.run_until_decided(comb, T0 + 300)
            (eid,) = comb.events
            url = f"http://127.0.0.1:{httpd.server_address[1]}/api/events/{eid}/review"
            req = urllib.request.Request(url, data=json.dumps(dict(
                verdict="frb", by="tester", note="clear sweep in three bands")).encode(),
                method="POST", headers={"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req))
            self.assertEqual(r["review"], "frb")
            with self.assertRaises(ValueError):
                comb.review(eid, "maybe")
            files = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(out, "events"))
                     for f in fs if f.endswith(".json")]
            with open(files[0]) as f:
                self.assertEqual(json.load(f)["reviews"][0]["by"], "tester")
        finally:
            httpd.shutdown()
            httpd.server_close()
            shutil.rmtree(out, ignore_errors=True)

    def test_unconfirmable_when_others_are_down_or_cannot_see(self):
        nodes = {n: FakeMonitor(n, b, followup_snr=9.0) for n, b in BANDS.items()}
        nodes["mid"].up = False
        nodes["lo"].up = False
        nodes["hi"].candidate("c1", 10.0, 120.0, T0, self.ra, self.dec)
        comb, out = make(nodes)
        try:
            self.run_until_decided(comb, T0 + 200)
            (ev,) = comb.events.values()
            self.assertEqual(ev["status"], "unconfirmable")
            self.assertEqual(ev["bands"]["mid"]["state"], "node_down")
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_position_setting_before_the_low_band(self):
        """A burst that reaches the lowest band only after its position has set"""
        ra, dec = None, None
        for r in np.arange(0, 360, 2.0):  # just above the western limit
            el, az = altaz(r, 10.0, T0)
            if 11 < el < 13 and 180 < az < 360:
                ra, dec = float(r), 10.0
                break
        dm = 2500.0  # 2.5 h of delay to 33 MHz
        nodes = {n: FakeMonitor(n, b, followup_snr=9.0) for n, b in BANDS.items()}
        nodes["hi"].candidate("c1", 20.0, dm, T0 - K_DM * dm / 74.27 ** 2 + 30, ra, dec)
        comb, out = make(nodes)
        try:
            comb.poll(now=T0 + 60)
            (ev,) = comb.events.values()
            self.assertEqual(ev["bands"]["lo"]["state"], "not_visible")
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_low_band_first_marks_passed_bands_as_searched(self):
        """A candidate first seen in the lowest band: the burst already passed the others"""
        dm = 300.0
        nodes = {n: FakeMonitor(n, b, followup_snr=9.0) for n, b in BANDS.items()}
        nodes["lo"].candidate("c1", 9.0, dm, T0, self.ra, self.dec)
        comb, out = make(nodes)
        try:
            t = T0 + K_DM * dm / 32.97 ** 2 + 60  # the low band reports a minute after arrival
            self.run_until_decided(comb, t)
            (ev,) = comb.events.values()
            for name in ("hi", "mid"):
                self.assertEqual(ev["bands"][name]["state"], "searched")
                # a watch is still sent, so the monitor keeps the accumulated spectrum
            self.assertTrue(ev["bands"][name]["watch_id"].endswith("-ctx"))
            self.assertEqual(ev["status"], "band_limited")
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_restart_reopens_events(self):
        nodes = {n: FakeMonitor(n, b) for n, b in BANDS.items()}
        nodes["hi"].candidate("c1", 10.0, 120.0, T0, self.ra, self.dec)
        comb, out = make(nodes)
        try:
            comb.poll(now=T0 + 100)
            comb2, _ = make(nodes)
            comb2.cfg.out_dir = out
            comb2 = Combiner(CombinerConfig(nodes=comb.cfg.nodes, out_dir=out))
            self.assertEqual(set(comb2.events), set(comb.events))
            self.assertIn(("hi", "c1"), comb2.seen)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_coordinated_injection_requested_from_every_band(self):
        nodes = {n: FakeMonitor(n, b) for n, b in BANDS.items()}
        comb, out = make(nodes, inject_interval_s=1.0, seed=1)
        try:
            comb.next_inject = T0
            comb.poll(now=T0 + 1)
            ids = {getattr(m, "injected", {}).get("id") for m in nodes.values()}
            self.assertEqual(len(ids), 1)
            spec = nodes["lo"].injected
            self.assertTrue(spec["id"].startswith("cinj-"))
            self.assertGreaterEqual(altaz(spec["ra"], spec["dec"], T0)[0], 30.0)
            with open(os.path.join(out, "injections.jsonl")) as f:
                self.assertEqual(json.loads(f.readline())["event"], "requested")
        finally:
            shutil.rmtree(out, ignore_errors=True)


class TestTwoBandsEndToEnd(unittest.TestCase):
    """Two real monitors fed the same FRBs, the combiner talking to them over HTTP"""

    def test_frb_confirmed_across_bands_with_followup(self):
        from skymon.geometry import Band
        from skymon.monitor import Config, Monitor
        from skymon.server import serve
        from skymon.simulate import SkySimulator, tier_dt_function

        out = tempfile.mkdtemp()
        start = T0
        ra, dec = zenith_ish(start + 200)
        dm = 40.0
        # one burst reaching the high band at start + 200 s
        frb = dict(id="sim-x", ra=ra, dec=dec, dm=dm, snr=25.0, width_s=0.003,
                   t_inf=start + 200 - K_DM * dm / 77.33 ** 2)
        bands = {"hi": 3104 - 64, "mid": 2400 - 64}  # 74.3-77.3 and 55.9-58.9 MHz
        sims, mons, servers = {}, {}, {}
        try:
            for name, chan0 in bands.items():
                b = Band(chan0, 64)
                sims[name] = SkySimulator(
                    chan0=chan0, nchan=64, start_unix=start, frb_per_hour=0, rfi_per_hour=0,
                    seed=hash(name) % 1000, frb_list=[dict(frb)],
                    tier_dt=tier_dt_function(b.edges_mhz(), 0.083592, 64, 300.0, 64))
                mons[name] = Monitor(Config(out_dir=os.path.join(out, name), nside=32,
                                            dm_max=300.0, block0=64, nthreads=4,
                                            warmup_images=32, inject_interval_s=0,
                                            accum_hours=0.2,
                                            # the mid band only measures: a high threshold
                                            threshold=8.0 if name == "hi" else 1e9))
                servers[name] = serve(mons[name], host="127.0.0.1", port=0)
            comb = Combiner(CombinerConfig(
                nodes={n: f"http://127.0.0.1:{s.server_address[1]}" for n, s in servers.items()},
                out_dir=os.path.join(out, "comb"), settle_s=0.0))
            n_img = int(330 / 0.083592)
            for i in range(n_img):
                for name in bands:
                    meta, img = sims[name].next_image()
                    mons[name].process(meta, img)
                if i % 60 == 0:
                    comb.poll(now=sims["hi"]._t(i))
                if i == int(150 / 0.083592):  # a burst that passed the high band a minute ago
                    mons["hi"].watches.add(dict(id="late", event_id="ev-late", ra=ra, dec=dec,
                                                dm=dm, t_inf=start + 90 - K_DM * dm / 74.25 ** 2,
                                                t_tol=2.0))
            for k in range(30):  # let the deadlines pass
                comb.poll(now=sims["hi"]._t(n_img) + 60 * k)
            evs = list(comb.events.values())
            self.assertEqual(len(evs), 1, [e["status"] for e in evs])
            ev = evs[0]
            self.assertEqual(ev["status"], "confirmed", json.dumps(ev["bands"], default=str))
            self.assertEqual(ev["bands"]["hi"]["state"], "detected")
            self.assertEqual(ev["bands"]["mid"]["state"], "followup")
            fu = ev["bands"]["mid"]["followup"]
            self.assertGreater(fu["snr"], 6.0)
            # the followup was stored with its data on the mid band's disk
            stored = [c for c in mons["mid"].store.list() if c.cls == "followup"]
            self.assertEqual(len(stored), 1)
            mons["mid"].store.flush()
            self.assertTrue(os.path.exists(stored[0].files["npz"]))
            # the late watch left the accumulated spectrum around its arrival
            ctx = [c for c in mons["hi"].store.list() if c.cls == "context"]
            self.assertEqual(len(ctx), 1)
            mons["hi"].store.flush()
            with np.load(ctx[0].files["npz"]) as z:
                self.assertGreater(len(z["accum_t"]), 5)
        finally:
            for s in servers.values():
                s.shutdown()
                s.server_close()
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
