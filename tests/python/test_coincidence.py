"""
Tests of the combiner's cross-band coincidence veto (src/python/skymon/combiner.py):
undispersed triggers in several bands at once are terrestrial, not bursts.

    cd tests/python && python -m unittest -v test_coincidence
"""
import json
import os
import shutil
import unittest

from test_combiner import BANDS, T0, FakeMonitor, make, zenith_ish

from skymon.combiner import Combiner, CombinerConfig
from skymon.dedisp import K_DM


class TestCoincidenceVeto(unittest.TestCase):
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

    def at_bottom(self, mon, cid, snr, dm, t_bottom, ra, dec, cls="candidate"):
        """A candidate whose arrival at the bottom of mon's band is t_bottom"""
        mon.candidate(cid, snr, dm, t_bottom - K_DM * dm / mon.band[0] ** 2, ra, dec, cls=cls)

    def test_simultaneous_in_two_bands_is_terrestrial(self):
        """DM 40 puts 74 and 56 MHz 23 s apart; triggers at the same moment are undispersed"""
        nodes = {n: FakeMonitor(n, b, followup_snr=0.0) for n, b in BANDS.items()}
        tb = T0 + 100
        self.at_bottom(nodes["hi"], "h1", 12.0, 40.0, tb, self.ra, self.dec)
        # imaged elsewhere in the other band, as interference in the near field is
        self.at_bottom(nodes["mid"], "m1", 9.0, 40.0, tb + 0.3, self.ra + 20, self.dec - 10)
        comb, out = make(nodes)
        try:
            self.run_until_decided(comb, tb + 60)
            terr = [e for e in comb.events.values() if e["kind"] == "terrestrial"]
            self.assertEqual(len(terr), 1)
            (ev,) = terr
            self.assertEqual(ev["status"], "terrestrial")
            self.assertEqual(ev["review"], "not needed")
            self.assertEqual(ev["nodes"], ["hi", "mid"])
            self.assertEqual({(c["node"], c["id"]) for c in ev["candidates"]},
                             {("hi", "h1"), ("mid", "m1")})
            for c in ev["candidates"]:
                for k in ("t_bottom", "dm", "snr"):
                    self.assertIn(k, c)
            self.assertIn("undispersed", ev["reason"])
            # no sky event is left for these candidates
            self.assertFalse([e for e in comb.events.values() if e["kind"] == "sky"
                              and e["status"] in ("pending", "confirmed")])
            st = comb.status()
            self.assertEqual(st["events"]["terrestrial"], 1)
            self.assertEqual(st["terrestrial"], dict(events=1, candidates=2))
            self.assertEqual(st["to_review"], 0)
            # stored, and found again after a restart
            files = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(out, "events"))
                     for f in fs if f.endswith(".json")]
            with open(files[0]) as f:
                self.assertEqual(json.load(f)["kind"], "terrestrial")
            comb2 = Combiner(CombinerConfig(nodes=comb.cfg.nodes, out_dir=out))
            self.assertEqual(comb2.terrestrial[("mid", "m1")], ev["id"])
            self.assertIn(("hi", "h1"), comb2.seen)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_rfi_in_another_band_moves_a_pending_sky_event_out(self):
        """A candidate opens a sky event; a simultaneous RFI trigger in another band follows"""
        nodes = {n: FakeMonitor(n, b, followup_snr=0.0) for n, b in BANDS.items()}
        tb = T0 + 100
        self.at_bottom(nodes["hi"], "h1", 12.0, 40.0, tb, self.ra, self.dec)
        comb, out = make(nodes)
        try:
            comb.poll(now=tb + 20)
            (sky,) = comb.events.values()
            self.assertEqual(sky["status"], "pending")
            self.at_bottom(nodes["lo"], "l1", 30.0, 2.5, tb - 0.5, self.ra, self.dec, cls="rfi")
            comb.poll(now=tb + 30)
            self.assertEqual(sky["status"], "terrestrial")
            self.assertTrue(any("terrestrial event" in n for n in sky["notes"]))
            (terr,) = [e for e in comb.events.values() if e["kind"] == "terrestrial"]
            self.assertEqual(terr["sky_events"], [sky["id"]])
            self.assertEqual(sky["terrestrial_event"], terr["id"])
            self.run_until_decided(comb, tb + 40)
            self.assertEqual(sky["status"], "terrestrial")  # not decided again as a burst
            self.assertIn(sky["id"], comb.events)  # the record is kept
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_dispersed_burst_is_confirmed_not_terrestrial(self):
        """A burst at DM 40 reaches 56 MHz about 23 s after 74 MHz"""
        dm = 40.0
        nodes = {n: FakeMonitor(n, b, followup_snr=0.0) for n, b in BANDS.items()}
        nodes["hi"].candidate("h1", 12.0, dm, T0, self.ra, self.dec)
        nodes["mid"].candidate("m1", 10.0, dm, T0, self.ra + 0.5, self.dec)
        gap = nodes["mid"].cands[0]["t_bottom"] - nodes["hi"].cands[0]["t_bottom"]
        self.assertGreater(gap, 20.0)
        comb, out = make(nodes)
        try:
            self.run_until_decided(comb, T0 + 200)
            (ev,) = comb.events.values()
            self.assertEqual(ev["kind"], "sky")
            self.assertEqual(ev["status"], "confirmed")
            self.assertEqual(comb.status()["events"]["terrestrial"], 0)
            self.assertFalse(comb.terrestrial)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_adjacent_bands_at_low_dm_are_not_vetoed(self):
        """At DM 3, 69.7 and 74.3 MHz are 0.3 s apart: near-simultaneity is what a burst shows"""
        dm = 3.0
        bands = {"hi": BANDS["hi"], "hi2": (69.67, 72.73)}
        nodes = {n: FakeMonitor(n, b, followup_snr=0.0) for n, b in bands.items()}
        nodes["hi"].candidate("h1", 12.0, dm, T0, self.ra, self.dec)
        nodes["hi2"].candidate("k1", 11.0, dm, T0, self.ra, self.dec)
        gap = nodes["hi2"].cands[0]["t_bottom"] - nodes["hi"].cands[0]["t_bottom"]
        self.assertLess(gap, 1.0)
        comb, out = make(nodes)
        try:
            self.run_until_decided(comb, T0 + 60)
            self.assertFalse([e for e in comb.events.values() if e["kind"] == "terrestrial"
                              or e["status"] == "terrestrial"])
            (ev,) = comb.events.values()
            self.assertEqual(ev["status"], "confirmed")
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_same_node_and_distant_times_are_not_coincidences(self):
        """Two triggers in one band, or in two bands a minute apart, are not vetoed"""
        nodes = {n: FakeMonitor(n, b, followup_snr=0.0) for n, b in BANDS.items()}
        tb = T0 + 100
        self.at_bottom(nodes["hi"], "h1", 12.0, 40.0, tb, self.ra, self.dec)
        self.at_bottom(nodes["hi"], "h2", 11.0, 40.0, tb + 0.2, self.ra + 30, self.dec)
        self.at_bottom(nodes["mid"], "m1", 10.0, 40.0, tb + 60, self.ra + 60, self.dec - 20)
        comb, out = make(nodes)
        try:
            comb.poll(now=tb + 70)
            self.assertFalse(comb.terrestrial)
            self.assertFalse([e for e in comb.events.values() if e["kind"] == "terrestrial"])
        finally:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
