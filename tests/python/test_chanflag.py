"""
Tests of the interference flagging of the all-sky monitor (skymon_normalise,
search.Searcher, monitor.Monitor).

    cd tests/python && python -m unittest -v test_chanflag

A channel that carries interference over much of the sky in one image is set
to zero for that image; a channel that does so persistently is masked until
the interference stops; neither enters the running statistics. The end-to-end
test adds a moving narrowband carrier to a simulated sky and checks that it
leaves no candidate behind while a simulated burst is still found.
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

from skymon import kernels  # noqa: E402
from skymon.dedisp import K_DM  # noqa: E402
from skymon.geometry import Band, altaz  # noqa: E402

T0 = 1792000000.0


class _State:
    """Normalisation state for kernels.normalise on synthetic [C, P] data"""

    def __init__(self, C, P, nsub=1024, warmup=32, seed=0, clip=12.0, max_hold=359):
        self.C, self.P = C, P
        self.mu = np.zeros((C, P), np.float32)
        self.var = np.zeros((C, P), np.float32)
        self.prev = np.zeros((C, P), np.float32)
        self.age = np.zeros(P, np.int32)
        self.live = np.ones(P, np.uint8)
        rng = np.random.default_rng(seed)
        self.sub = np.sort(rng.choice(P, min(nsub, P), replace=False)).astype(np.int32)
        self.med = np.zeros(C, np.float32)
        self.flags = np.zeros(C, np.uint8)
        self.spread = np.zeros(C, np.float32)
        self.held = np.zeros((C, P), np.uint16)
        self.warmup, self.clip, self.max_hold = warmup, clip, max_hold

    def copy(self):
        s = _State.__new__(_State)
        for k, v in self.__dict__.items():
            setattr(s, k, v.copy() if isinstance(v, np.ndarray) else v)
        return s

    def push(self, raw, mask=None):
        dt = 0.0836
        kernels.normalise(raw, self.mu, self.var, self.prev, self.age, self.live, None, self.sub,
                          self.warmup, dt / 120.0, dt / 300.0, self.clip, self.med, 2,
                          chan_flag_sigma=1.6, chan_mask=mask, flags=self.flags,
                          spread=self.spread, held=self.held, max_hold=self.max_hold)
        return raw


def _sky(rng, C, P, level):
    return (level + rng.standard_normal((C, P))).astype(np.float32)


def _interference(rng, P, frac=0.6, amp=10.0):
    """A carrier seen over frac of the sky, its strength varying across it (sidelobes)"""
    on = rng.random(P) < frac
    return (on * np.abs(rng.normal(0.0, amp, P))).astype(np.float32)


class TestKernelFlagging(unittest.TestCase):
    C, P = 32, 3000

    def setUp(self):
        self.rng = np.random.default_rng(11)
        # every pixel and channel its own sky level, so the flagging works on
        # normalised values and not on raw ones
        self.level = self.rng.uniform(5.0, 50.0, (self.C, self.P)).astype(np.float32)
        self.st = _State(self.C, self.P)
        for _ in range(200):
            self.st.push(_sky(self.rng, self.C, self.P, self.level))

    def test_channel_with_interference_is_flagged_and_zeroed(self):
        raw = _sky(self.rng, self.C, self.P, self.level)
        ref_state = self.st.copy()
        ref = ref_state.push(raw.copy())
        bad = raw.copy()
        bad[5] += _interference(self.rng, self.P)
        before = (self.st.mu[5].copy(), self.st.var[5].copy(), self.st.prev[5].copy())
        out = self.st.push(bad)
        self.assertEqual(np.nonzero(self.st.flags)[0].tolist(), [5])
        self.assertTrue(np.all(out[5] == 0))
        self.assertGreater(self.st.spread[5], 3.0)
        # the other channels are exactly as without the interference
        others = np.arange(self.C) != 5
        np.testing.assert_array_equal(out[others], ref[others])
        self.assertEqual(int(np.count_nonzero(ref_state.flags)), 0)
        # and the flagged channel did not enter the statistics
        np.testing.assert_array_equal(self.st.mu[5], before[0])
        np.testing.assert_array_equal(self.st.var[5], before[1])
        np.testing.assert_array_equal(self.st.prev[5], before[2])
        self.assertTrue(np.all(self.st.held[5] == 1))
        # the next clean image releases the hold and is not flagged
        self.st.push(_sky(self.rng, self.C, self.P, self.level))
        self.assertEqual(int(np.count_nonzero(self.st.flags)), 0)
        self.assertEqual(int(self.st.held[5].max()), 0)

    def test_noise_is_not_flagged(self):
        n_flag = 0
        spreads = []
        for _ in range(1000):
            self.st.push(_sky(self.rng, self.C, self.P, self.level))
            n_flag += int(np.count_nonzero(self.st.flags))
            spreads.append(self.st.spread.copy())
        # false flag rate per channel and image below 0.1%: none at all here
        self.assertLess(n_flag / (1000 * self.C), 1e-3)
        self.assertEqual(n_flag, 0)
        s = np.concatenate(spreads)
        self.assertLess(abs(float(np.median(s)) - 1.0), 0.1)
        self.assertLess(float(s.max()), 1.4)

    def test_clipped_cells_are_zeroed_and_withheld(self):
        raw = _sky(self.rng, self.C, self.P, self.level)
        raw[3, 17] += 30.0
        mu = self.st.mu[3, 17]
        out = self.st.push(raw)
        self.assertEqual(out[3, 17], 0.0)
        self.assertEqual(self.st.mu[3, 17], mu)
        self.assertEqual(int(self.st.held[3, 17]), 1)
        self.assertEqual(int(np.count_nonzero(self.st.flags)), 0)

    def test_lasting_change_is_followed_after_the_hold(self):
        """A cell that stays far off is updated again after max_hold images"""
        st = self.st
        st.max_hold = 20
        for _ in range(21):
            raw = _sky(self.rng, self.C, self.P, self.level)
            raw[2, 9] += 40.0
            st.push(raw)
        mu = float(st.mu[2, 9])
        raw = _sky(self.rng, self.C, self.P, self.level)
        raw[2, 9] += 40.0
        st.push(raw)
        self.assertGreater(float(st.mu[2, 9]), mu)


class TestPersistentMask(unittest.TestCase):
    def test_masked_while_present_and_released_after(self):
        from skymon.search import Searcher

        rng = np.random.default_rng(12)
        edges = Band(2336, 32).edges_mhz()
        dt = 0.0836
        C, P, ch = 32, 400, 7
        se = Searcher(edges, dt, P, dm_max=40.0, block0=64, threshold=8.0, nthreads=2,
                      warmup_images=32, rng=rng)
        se.set_live(np.ones(P, bool))
        level = rng.uniform(5.0, 50.0, (C, P)).astype(np.float32)

        def run(n, frac_images):
            masked, zero_ok = [], True
            for _ in range(n):
                raw = _sky(rng, C, P, level)
                was_masked = bool(se.chan_mask[ch])
                if rng.random() < frac_images:
                    raw[ch] += _interference(rng, P)
                for _ in se.push(raw):
                    pass
                se.finish_block()
                if was_masked and np.any(raw[ch] != 0):
                    zero_ok = False
                masked.append(bool(se.chan_mask[ch]))
            return np.array(masked), zero_ok

        run(100, 0.0)
        self.assertEqual(int(se.chan_mask.sum()), 0)
        n_on = int(150 / dt)
        masked, zero_ok = run(n_on, 0.8)
        self.assertTrue(masked.any(), "never masked")
        first = int(np.argmax(masked))
        # masked within about a time constant (rate 0.8 reaches 0.5 after 0.98 tau)
        self.assertLess(first * dt, 45.0)
        self.assertTrue(masked[first:].all(), "released while the interference went on")
        self.assertTrue(zero_ok, "a masked channel was not set to zero")
        self.assertEqual(se.rfi_status()["masked_channels"], [ch])
        self.assertGreater(se.rfi_status()["flagged_fraction_last_minute"], 0.9 / C)
        # no other channel was ever flagged
        tot = se.flag_total.copy()
        tot[ch] = 0
        self.assertEqual(int(tot.sum()), 0)
        # the interference did not drag the statistics of its channel
        dev = (se.mu[ch] - level[ch]) / np.sqrt(se.var[ch])
        self.assertLess(float(np.abs(np.median(dev))), 0.3)
        # after it stops: released (0.8 falls to 0.2 in ln 4 = 1.4 tau)
        masked, _ = run(int(90 / dt), 0.0)
        self.assertFalse(masked[-1], "not released")
        last = int(np.nonzero(masked)[0].max()) if masked.any() else 0
        self.assertGreater(last * dt, 20.0)  # hysteresis: not released at once
        self.assertEqual(se.rfi_status()["masked_channels"], [])


def _frb_position(start):
    for r in range(0, 360, 3):
        for d in (20.0, 34.0, 50.0):
            if altaz(r, d, start + 200.0)[0] > 65 and altaz(r, d, start + 260.0)[0] > 65:
                return float(r), d
    raise RuntimeError("no position")


class Carrier:
    """
    A narrowband carrier scattered by a moving target (an aircraft, a meteor
    trail): one channel, a beam-sized peak that moves across the sky, and a
    sidelobe pattern over the whole image that moves with it. It comes in
    pings of 0.5-6 s with fading from image to image, separated by gaps of
    1-8 s, between t_on and t_off.
    """

    def __init__(self, chan, t_on, t_off, peak=40.0, sidelobe=8.0, seed=0):
        self.chan, self.t_on, self.t_off = chan, t_on, t_off
        self.peak, self.sidelobe = peak, sidelobe
        self.rng = np.random.default_rng(seed)
        self.next_change = t_on
        self.active = False
        self.n_on = 0

    def apply(self, sim, img, t):
        if t < self.t_on or t > self.t_off:
            return
        if t >= self.next_change:
            self.active = not self.active
            self.next_change = t + (self.rng.uniform(0.5, 6.0) if self.active
                                    else self.rng.uniform(1.0, 8.0))
        if not self.active:
            return
        self.n_on += 1
        G = img.shape[-1]
        f = (t - self.t_on) / (self.t_off - self.t_on)
        x0, y0 = 0.15 * G + 0.7 * G * f, 0.3 * G + 0.4 * G * f  # moves across the image
        yy, xx = np.mgrid[0:G, 0:G]
        fade = self.rng.rayleigh(0.8)
        pk = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 2.0 ** 2))
        lobes = np.cos(2 * np.pi * (xx - x0) / 17.0) * np.cos(2 * np.pi * (yy - y0) / 23.0)
        sig = sim.sigma[self.chan]
        img[self.chan] += (fade * (self.peak * pk + self.sidelobe * lobes) * sig).astype(np.float32)


class TestEndToEndCarrier(unittest.TestCase):
    def test_carrier_leaves_no_candidates_and_burst_is_found(self):
        from skymon.evaluate import report
        from skymon.monitor import Config, Monitor
        from skymon.simulate import SkySimulator, tier_dt_function

        out = tempfile.mkdtemp()
        try:
            chan0, nchan = 2336, 64
            edges = Band(chan0, nchan).edges_mhz()
            ra, dec = _frb_position(T0)
            dm = 45.0
            # the burst crosses the band during the carrier, 230 s after the start
            frb = dict(id="sim-frb", ra=ra, dec=dec, dm=dm, snr=25.0, width_s=0.005,
                       t_inf=T0 + 230.0 - K_DM * dm / edges[0] ** 2)
            sim = SkySimulator(chan0=chan0, nchan=nchan, start_unix=T0, frb_per_hour=0,
                               rfi_per_hour=0, truth_path=os.path.join(out, "truth.jsonl"),
                               seed=7, el_min=25.0, frb_list=[frb],
                               tier_dt=tier_dt_function(edges, 0.083592, nchan, 300.0, 64))
            mon = Monitor(Config(out_dir=out, nside=32, dm_max=300.0, block0=64, nthreads=4,
                                 warmup_images=32, inject_interval_s=0, seed=7,
                                 accum_hours=0.5))
            carrier = Carrier(chan=40, t_on=T0 + 60.0, t_off=T0 + 300.0, seed=3)
            masked_seen = False
            for _ in range(int(340 / sim.dt)):
                meta, img = sim.next_image()
                carrier.apply(sim, img, meta.unix)
                mon.process(meta, img)
                masked_seen |= bool(mon.searcher.chan_mask[40])
            mon.store.flush()
            self.assertEqual([t["id"] for t in sim.truth], ["sim-frb"])
            self.assertGreater(carrier.n_on, 500)
            rep = report(os.path.join(out, "truth.jsonl"), out, t_end=sim._t(sim.n),
                         latency_s=30)
            # nothing but the burst is stored as a candidate
            self.assertEqual(len(rep["false"]), 0, rep["text"] + "\n" + "\n".join(
                f"{c['cls']} S/N {c['snr']:.1f} DM {c['dm']:.1f}" for c in rep["false"]))
            (t, c), = rep["rows"]
            self.assertIsNotNone(c, rep["text"])
            # 0.68-0.72 for this burst with or without the flagging; one burst
            # varies more than the median of TestEndToEnd (above 0.7)
            self.assertGreater(c["snr"] / t["snr"], 0.6, rep["text"])
            # the carrier's channel was flagged, the others hardly ever
            tot = mon.searcher.flag_total
            self.assertGreater(int(tot[40]), carrier.n_on // 2)
            others = np.delete(tot, 40)
            self.assertLess(int(others.sum()), 0.001 * len(others) * mon.searcher.images)
            st = mon.status()["rfi_flagging"]
            self.assertIn("flagged_fraction_last_minute", st)
            self.assertIn("masked_channels", st)
            q = c.get("quality") or {}
            self.assertIn("flagged_fraction", q)
            self.assertLess(q["flagged_fraction"], 0.1)
            print(f"\n{rep['text']}\ncarrier images {carrier.n_on}, flagged {int(tot[40])}, "
                  f"masked at some time {masked_seen}, status {st}", flush=True)
        finally:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
