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

    def test_tier_cascade_matches_single_pushes(self):
        """One cascade call does what successive tier_push calls and the decimation do"""
        from types import SimpleNamespace as NS
        rng = np.random.default_rng(6)
        C, P, W, n = 4, 37, 9, 3
        k = float(np.float32(1 / math.sqrt(2)))

        def tiers():
            return [NS(base=rng.standard_normal((C, P)).astype(np.float32),
                       history=np.zeros((C, W, P), np.float32), pos=2 + j,
                       acc=rng.standard_normal((C, P)).astype(np.float32)) for j in range(n)]
        a = tiers()
        b = [NS(base=t.base.copy(), history=t.history.copy(), pos=t.pos, acc=t.acc.copy())
             for t in a]
        x = rng.standard_normal((C, P)).astype(np.float32)
        kernels.tier_cascade(x, a, 0.03, k, last_dec=True, nthreads=2)
        y = x
        for j, t in enumerate(b):
            kernels.tier_push(y, t.base, t.history, t.pos, 0.03, t.acc)
            if j + 1 < n:
                y = t.acc * np.float32(k)
                t.acc[:] = 0
        for u, v in zip(a, b):
            for f in ("base", "history", "acc"):
                np.testing.assert_array_equal(getattr(u, f), getattr(v, f))

    def test_search_matches_reference(self):
        """kernels.search: the FDMT and boxcar search, against numpy on the same transform"""
        rng = np.random.default_rng(8)
        C, D, W, nb, P, seg = 64, 128, 200, 64, 45, 16
        plan = fdmt_plan(self.edges, D)
        flat = kernels.flatten_plan(plan)
        x = rng.standard_normal((C, W, P)).astype(np.float32)
        x[:, W - 20, 5] += 3.0
        widths = np.array([1, 2, 4, 8], np.int32)
        sigma = (8 + rng.random((D + 1, 4))).astype(np.float32)
        for row0 in (0, D // 2 + 1):
            best, info, sumsq, cnt = kernels.search(x, flat, D, row0, nb, widths, sigma, seg,
                                                    clip=5.0, nthreads=3, ptile=8, detrend=False)
            rows = fdmt(x, plan)  # [D + 1, W, P]
            # prefix sums from the first sample of the widest boxcar
            t0 = W - nb + 1 - int(widths.max())
            cs = np.zeros((D + 1, W + 1, P), np.float32)
            cs[:, t0 + 1:] = np.cumsum(rows[:, t0:], axis=1, dtype=np.float32)
            nseg = nb // seg
            ref = np.full((P, nseg), -np.inf, np.float32)
            rinfo = np.full((P, nseg, 3), -1, np.int32)
            ssq = np.zeros((D + 1, 4))
            for d in range(row0, D + 1):
                for wi, w in enumerate(widths):
                    inv = np.float32(1) / sigma[d, wi]
                    for t in range(W - nb, W):
                        v = cs[d, t + 1] - cs[d, t + 1 - w]
                        ssq[d, wi] += np.sum(v[np.abs(v) < 5 * sigma[d, wi]].astype(float) ** 2)
                        snr = v * inv
                        sg = (t - (W - nb)) // seg
                        up = snr > ref[:, sg]
                        ref[up, sg] = snr[up]
                        rinfo[up, sg] = (d, t, wi)
            np.testing.assert_array_equal(best, ref)
            np.testing.assert_array_equal(info, rinfo)
            np.testing.assert_allclose(sumsq, ssq, rtol=1e-5)

    def test_window_shift(self):
        h = np.random.default_rng(2).standard_normal((3, 10, 5)).astype(np.float32)
        ref = h.copy()
        ref[:, :4] = h[:, 6:10]
        kernels.shift(h, 0, 6, 4, nthreads=2)
        np.testing.assert_array_equal(h, ref)
        h2 = ref.copy()
        ref2 = ref.copy()
        ref2[:, :7] = ref[:, 3:10].copy()
        kernels.shift(h2, 0, 3, 7, nthreads=2)  # overlapping
        np.testing.assert_array_equal(h2, ref2)

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


BAND192 = Band(chan0=3072, nchan=192)  # 73.49-78.08 MHz: all channels of an NDP stream


def _edges(C, f_lo=73.4880, width=4.5938):
    return f_lo + np.arange(C + 1) * (width / C)


class TestAnyNumberOfChannels(unittest.TestCase):
    """
    The FDMT for any number of channels (192 for the FX imager's streams): a
    level with an odd number of subbands passes its lowest subband on
    unmerged, and nothing is padded
    """

    def test_plan_of_192_channels(self):
        init, levels = fdmt_plan(BAND192.edges_mhz(), 384)
        self.assertEqual(len(init), 192)
        self.assertEqual([len(level) for level in levels], [96, 48, 24, 12, 6, 3, 2, 1])
        # three subbands of 64: the lowest is passed on and merged last
        self.assertIsNone(levels[6][0][1])
        self.assertIsNotNone(levels[6][1][1])
        self.assertEqual(levels[-1][0][0], 384)
        # a power of two has no pass
        _, lv = fdmt_plan(BAND.edges_mhz(), 128)
        self.assertTrue(all(d_lo is not None for level in lv for _, d_lo, _ in level))

    def test_kernel_matches_numpy_for_any_number_of_channels(self):
        rng = np.random.default_rng(21)
        for C in (1, 2, 3, 5, 6, 7, 12, 96, 192):
            D = 2 * C
            plan = fdmt_plan(_edges(C), D)
            x = rng.standard_normal((C, 40 + D, 5)).astype(np.float32)
            ref = fdmt(x, plan)
            got = kernels.fdmt(x, kernels.flatten_plan(plan), D, nthreads=3, ptile=4)
            self.assertEqual(ref.shape, (D + 1, 40 + D, 5))
            np.testing.assert_array_equal(ref, got, err_msg=f"{C} channels")

    def test_search_matches_reference_with_192_channels(self):
        """kernels.search, all rows or only those of a tier above 0, against numpy"""
        rng = np.random.default_rng(22)
        C, D, W, nb, P, seg = 192, 384, 420, 32, 9, 16
        plan = fdmt_plan(BAND192.edges_mhz(), D)
        flat = kernels.flatten_plan(plan)
        x = rng.standard_normal((C, W, P)).astype(np.float32)
        widths = np.array([1, 2, 4, 8], np.int32)
        sigma = (8 + rng.random((D + 1, 4))).astype(np.float32)
        rows = fdmt(x, plan)
        t0 = W - nb + 1 - int(widths.max())  # the first sample a boxcar reads
        cs = np.zeros((D + 1, W + 1, P), np.float32)
        cs[:, t0 + 1:] = np.cumsum(rows[:, t0:], axis=1, dtype=np.float32)
        for row0 in (0, D // 2 + 1, D):
            best, info, _, _ = kernels.search(x, flat, D, row0, nb, widths, sigma, seg,
                                              clip=5.0, nthreads=3, ptile=8, detrend=False)
            nseg = nb // seg
            ref = np.full((P, nseg), -np.inf, np.float32)
            rinfo = np.full((P, nseg, 3), -1, np.int32)
            for d in range(row0, D + 1):
                for wi, w in enumerate(widths):
                    inv = np.float32(1) / sigma[d, wi]
                    for t in range(W - nb, W):
                        snr = (cs[d, t + 1] - cs[d, t + 1 - w]) * inv
                        sg = (t - (W - nb)) // seg
                        up = snr > ref[:, sg]
                        ref[up, sg] = snr[up]
                        rinfo[up, sg] = (d, t, wi)
            np.testing.assert_array_equal(best, ref)
            np.testing.assert_array_equal(info, rinfo)

    def test_pulse_keeps_its_fluence_in_192_channels(self):
        """
        A dispersed pulse is summed at its delay with the same fraction of its
        fluence in 192 channels as in 128: the unequal last merge loses nothing
        """
        dt = 0.0418
        frac = {}
        for band in (Band(3104, 128), BAND192):
            e = band.edges_mhz()
            C, D = band.nchan, 2 * band.nchan
            plan = fdmt_plan(e, D)
            s = K_DM * (e[0] ** -2 - e[-1] ** -2)
            got = []
            for f in (0.3, 0.55, 0.9):
                for phase in (0.0, 0.25, 0.5, 0.75):
                    delay = int(f * D)
                    dm = delay * dt / s
                    dyn = dispersed(e, dt, D + 80, dm, t_top=(40 + phase) * dt)
                    out = fdmt(dyn[:, :, None].astype(np.float32), plan)[:, :, 0]
                    d, t = np.unravel_index(np.argmax(out), out.shape)
                    self.assertLessEqual(abs(d - delay), 1, (C, delay, d))
                    got.append(float(out.max()) / C)  # the fraction of the fluence summed
            frac[C] = np.array(got)
        self.assertTrue(np.all(frac[192] > 0.8), frac)
        self.assertAlmostEqual(frac[192].mean(), frac[128].mean(), delta=0.01, msg=str(frac))

    def test_searcher_uses_the_real_band_edges(self):
        from skymon.search import Searcher
        se = Searcher(BAND192.edges_mhz(), 0.0418, 8, dm_max=3000.0, max_delay=384, block0=512,
                      nthreads=1)
        self.assertEqual(se.C, 192)
        self.assertAlmostEqual(se.f_lo, 3071.5 * 196 / 8192, places=6)
        self.assertAlmostEqual(se.f_hi, 3263.5 * 196 / 8192, places=6)
        # the delay rows span the real band: tier 0 reaches DM 183 at 41.8 ms
        self.assertAlmostEqual(se.tiers[0].dm_max, 182.9, delta=0.2)

    def test_snr_grows_as_the_square_root_of_the_channels(self):
        """
        The same bursts per channel, searched in 128 and in 192 channels of the
        same width: S/N in the ratio sqrt(192 / 128), as for a full band
        """
        from skymon.search import Searcher
        dt, P = 0.0418, 16
        snr = {}
        for band in (Band(3104, 128), BAND192):
            rng = np.random.default_rng(23)
            e = band.edges_mhz()
            se = Searcher(e, dt, P, dm_max=40.0, block0=64, max_delay=2 * band.nchan,
                          threshold=6.0, nthreads=2, warmup_images=16, rng=rng,
                          widths=(1, 2, 4, 8))
            se.set_live(np.ones(P, bool))
            dm = 20.0
            # eight bursts at eight arrival phases, in every other pixel
            bursts = [Burst("x", 0, 0, dm, 70.0 + k * dt / 8 - K_DM * dm / e[-1] ** 2,
                            width_s=0.0) for k in range(8)]
            best = np.zeros(P)
            for n in range(int(140 / dt)):
                raw = rng.standard_normal((band.nchan, P)).astype(np.float32) + 10.0
                for k, b in enumerate(bursts):
                    raw[:, 2 * k] += 3.0 * channel_fractions(b, e, n * dt, dt)
                for res in se.push(raw):
                    for h in res.hits:
                        best[h.slot] = max(best[h.slot], h.snr)
                se.finish_block()
            self.assertTrue(np.all(best[::2] > 10), best)
            snr[band.nchan] = best[::2].mean()
        self.assertAlmostEqual(snr[192] / snr[128], math.sqrt(1.5), delta=0.08, msg=str(snr))


class TestHalfPrecisionHistory(unittest.TestCase):
    """Tier histories in float16: the sums are formed in single precision"""

    def test_search_of_half_equals_search_of_its_single_values(self):
        rng = np.random.default_rng(24)
        for C, D in ((64, 128), (192, 384)):
            e = _edges(C)
            flat = kernels.flatten_plan(fdmt_plan(e, D))
            W, nb, P = D + 64, 64, 21
            h = (3 * rng.standard_normal((C, W, P))).astype(np.float16)
            sigma = (5 + rng.random((D + 1, 3))).astype(np.float32)
            for detrend, zerodm in ((2, False), (1, True)):
                a = kernels.search(h, flat, D, D // 2 + 1, nb, [1, 4, 16], sigma, 16,
                                   nthreads=3, detrend=detrend, zerodm=zerodm)
                b = kernels.search(h.astype(np.float32), flat, D, D // 2 + 1, nb, [1, 4, 16],
                                   sigma, 16, nthreads=3, detrend=detrend, zerodm=zerodm)
                for u, v in zip(a, b):
                    np.testing.assert_array_equal(u, v)

    def test_cascade_and_shift_in_half(self):
        from types import SimpleNamespace as NS
        rng = np.random.default_rng(25)
        C, P, W, n = 5, 37, 9, 3
        k = float(np.float32(1 / math.sqrt(2)))
        a_hp = np.array([1 / 32, 1 / 64, 1 / 128], np.float32)
        a = [NS(base=rng.standard_normal((C, P)).astype(np.float32),
                history=np.zeros((C, W, P), np.float32), pos=2 + j,
                acc=rng.standard_normal((C, P)).astype(np.float32)) for j in range(n)]
        b = [NS(base=t.base.copy(), history=t.history.astype(np.float16), pos=t.pos,
                acc=t.acc.copy()) for t in a]
        for _ in range(3):
            x = (10 * rng.standard_normal((C, P))).astype(np.float32)
            kernels.tier_cascade(x, a, a_hp, k, last_dec=True, nthreads=2)
            kernels.tier_cascade(x, b, a_hp, k, last_dec=True, nthreads=2)
            for u, v in zip(a, b):
                u.pos += 1
                v.pos += 1
        for u, v in zip(a, b):
            np.testing.assert_array_equal(u.base, v.base)
            np.testing.assert_array_equal(u.acc, v.acc)
            np.testing.assert_array_equal(u.history.astype(np.float16), v.history)
        h = b[0].history.copy()
        kernels.shift(b[0].history, 0, 4, 5, nthreads=2)
        np.testing.assert_array_equal(b[0].history[:, :5], h[:, 4:9])
        np.testing.assert_array_equal(b[0].history[:, 5:], h[:, 5:])
        with self.assertRaises(ValueError):
            kernels.shift(b[0].history.astype(np.float64), 0, 1, 1)

    def test_searcher_in_half_finds_the_same_burst(self):
        from skymon.search import Searcher
        dt, P = 0.0836, 16
        e = BAND.edges_mhz()
        found = {}
        for dtype in ("float32", "float16"):
            rng = np.random.default_rng(26)
            se = Searcher(e, dt, P, dm_max=150.0, block0=64, threshold=7.0, nthreads=2,
                          warmup_images=16, rng=rng, history_dtype=dtype)
            self.assertEqual(se.tiers[0].history.dtype, np.dtype(dtype))
            se.set_live(np.ones(P, bool))
            dm, t_top = 60.0, 150.0
            b = Burst("x", 0, 0, dm, t_top - K_DM * dm / e[-1] ** 2, width_s=0.0)
            hits = []
            for n in range(int(260 / dt)):
                raw = rng.standard_normal((BAND.nchan, P)).astype(np.float32) + 10.0
                raw[:, 7] += 2.0 * channel_fractions(b, e, n * dt, dt)
                for res in se.push(raw):
                    hits += res.hits
                se.finish_block()
            found[dtype] = max(hits, key=lambda h: h.snr)
            mem = se.memory_bytes()
            found[dtype + "_mem"] = sum(t.history.nbytes for t in se.tiers)
        a, b = found["float32"], found["float16"]
        self.assertEqual((a.slot, a.row, a.tier, a.t_end, a.width), (b.slot, b.row, b.tier,
                                                                     b.t_end, b.width))
        self.assertAlmostEqual(b.snr / a.snr, 1.0, delta=2e-3)
        self.assertEqual(found["float16_mem"] * 2, found["float32_mem"])
        self.assertGreater(mem, 0)


class TestCompiledIngest(unittest.TestCase):
    """The vector paths of the regridding and the normalisation against references"""

    def test_regrid_matches_reference(self):
        rng = np.random.default_rng(27)
        for C, G, P in ((192, 64, 999), (5, 32, 77)):
            img = rng.standard_normal((C, G, G)).astype(np.float32)
            xy_a = rng.uniform(-1, G + 1, (P, 2))
            xy_a[::40] = np.nan  # free slots: zero weights
            xy, idx = np.zeros((P, 2)), np.zeros((P, 16), np.int32)
            w = np.zeros((P, 16), np.float32)
            kernels.taps(xy_a, xy_a, 0.0, G, xy, idx, w, nthreads=2)
            # a few pixels with taps that are not a 4 x 4 block take the general loop
            idx[3] = rng.integers(0, G * G, 16)
            idx[10] = idx[10][::-1]
            out = np.full((C, P), np.nan, np.float32)
            kernels.regrid(img, idx, w, out, nthreads=3)
            ref = np.einsum("pk,cpk->cp", w.astype(np.float64), img.reshape(C, -1)[:, idx])
            np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
            self.assertTrue(np.all(out[:, ::40] == 0))
            # other tap counts (bilinear)
            i4, w4 = bilinear(rng.uniform(2, G - 2, P), rng.uniform(2, G - 2, P), G)
            out4 = np.zeros((C, P), np.float32)
            kernels.regrid(img, i4.astype(np.int32), w4.astype(np.float32), out4, nthreads=2)
            ref4 = np.einsum("pk,cpk->cp", w4.astype(np.float64), img.reshape(C, -1)[:, i4])
            np.testing.assert_allclose(out4, ref4, rtol=1e-5, atol=1e-5)

    def _norm_state(self, C, P, rng):
        st = dict(mu=rng.standard_normal((C, P)).astype(np.float32),
                  var=(0.5 + rng.random((C, P))).astype(np.float32),
                  prev=rng.standard_normal((C, P)).astype(np.float32),
                  age=np.full(P, 1000, np.int32), held=np.zeros((C, P), np.uint16),
                  acc=np.zeros((C, P), np.float32))
        st["var"][0, :3] = 0.0  # no variance yet: output 0
        return st

    def _normalise(self, raw, st, live, sub, nbr):
        C = raw.shape[0]
        med, flags = np.zeros(C, np.float32), np.zeros(C, np.uint8)
        kernels.normalise(raw, st["mu"], st["var"], st["prev"], st["age"], live, st["acc"], sub,
                          16, 0.01, 0.003, 6.0, med, 3, chan_flag_sigma=1.6,
                          chan_mask=np.zeros(C, np.uint8), flags=flags,
                          spread=np.zeros(C, np.float32), held=st["held"], max_hold=10, nbr=nbr)
        return med, flags

    def test_normalise_vector_and_scalar_paths_agree(self):
        """
        The same cells through the vector path (groups of 8 live pixels) and
        through the scalar path (each group broken by a pixel that is not
        live) give the same output and statistics, bit for bit; and both
        agree with a numpy reference
        """
        rng = np.random.default_rng(28)
        C, P = 24, 160
        st_a = self._norm_state(C, P, rng)
        raws = [(st_a["mu"] + rng.standard_normal((C, P)) * np.sqrt(st_a["var"]))
                .astype(np.float32) for _ in range(4)]
        raws[2][:, 50] += 40.0  # beyond the clip
        # layout b: a pixel that is not live before every 7 of layout a
        pos = np.arange(P) + np.arange(P) // 7 + 1
        Pb = int(pos[-1]) + 1
        live_a, live_b = np.ones(P, np.uint8), np.zeros(Pb, np.uint8)
        live_b[pos] = 1
        st_b = {}
        for k, v in st_a.items():
            if v.ndim == 2:
                st_b[k] = np.zeros((C, Pb), v.dtype)
                st_b[k][:, pos] = v
            else:
                st_b[k] = np.zeros(Pb, v.dtype)
                st_b[k][pos] = v
        st_b["age"][:] = np.where(live_b, 1000, 0)
        sub_a = np.arange(0, P, 2, dtype=np.int32)
        sub_b = pos[sub_a].astype(np.int32)
        nbr_a = np.full((P, 8), -1, np.int32)
        nbr_b = np.full((Pb, 8), -1, np.int32)
        for k, raw in enumerate(raws):
            ref_mu, ref_var = st_a["mu"].copy(), st_a["var"].copy()
            ref_raw = raw.copy()
            ra = raw.copy()
            rb = np.zeros((C, Pb), np.float32)
            rb[:, pos] = raw
            med_a, _ = self._normalise(ra, st_a, live_a, sub_a, nbr_a)
            med_b, _ = self._normalise(rb, st_b, live_b, sub_b, nbr_b)
            np.testing.assert_array_equal(med_a, med_b)
            np.testing.assert_array_equal(ra, rb[:, pos])
            self.assertTrue(np.all(rb[:, live_b == 0] == 0))
            for f in ("mu", "var", "prev", "held", "acc"):
                np.testing.assert_array_equal(st_a[f], st_b[f][:, pos], err_msg=f)
            # numpy reference of the steady state (no flag; one clipped cell)
            dev = np.where(ref_var > 0, (ref_raw - ref_mu) / np.sqrt(np.maximum(ref_var, 1e-30)),
                           0).astype(np.float32)
            y = dev - med_a[:, None]
            out = np.where(np.abs(y) > 6.0, 0, y)
            np.testing.assert_allclose(ra, out, rtol=1e-5, atol=1e-5)
            self.assertEqual(k == 2, bool(np.all(ra[:, 50] == 0)))


class TestClusterHits(unittest.TestCase):
    @staticmethod
    def _reference(hits, ra, dec, radius_deg, d_rows):
        """The original loop over group heads, for comparison"""
        from skymon.candidates import unit_vectors
        order = np.argsort([-h.snr for h in hits])
        vec = unit_vectors(ra, dec)
        cos_r = np.cos(np.radians(np.broadcast_to(np.asarray(radius_deg, float), (len(hits),))))
        heads, groups = [], []
        for i in order:
            h = hits[i]
            for g, k in enumerate(heads):
                b = hits[k]
                if abs(h.row - b.row) > d_rows:
                    continue
                if abs(h.t_end - b.t_end) > abs(h.row - b.row) + h.width + b.width + 2:
                    continue
                if float(vec[i] @ vec[k]) < cos_r[k]:
                    continue
                groups[g].append(int(i))
                break
            else:
                heads.append(i)
                groups.append([int(i)])
        return groups

    def test_same_groups_as_the_loop(self):
        """The vectorised clustering joins each hit to the same (first matching) head"""
        from types import SimpleNamespace
        from skymon.candidates import cluster_hits
        rng = np.random.default_rng(5)
        for n, spread in ((1, 1), (50, 30), (3000, 4), (3000, 60)):
            slot = rng.integers(0, 500, n)
            pos_ra, pos_dec = rng.uniform(0, 30, 500), rng.uniform(-10, 20, 500)
            hits = [SimpleNamespace(slot=int(s), row=int(r), t_end=int(t), width=int(w),
                                    snr=float(round(8 + rng.exponential(3), 1)))  # ties
                    for s, r, t, w in zip(slot, rng.integers(0, spread, n),
                                          rng.integers(0, 40, n), rng.integers(1, 9, n))]
            radius = rng.uniform(0.5, 4, n)  # one per hit
            ra, dec = pos_ra[slot], pos_dec[slot]
            self.assertEqual(cluster_hits(hits, ra, dec, radius, 16),
                             self._reference(hits, ra, dec, radius, 16))
            self.assertEqual(cluster_hits(hits, ra, dec, 2.0, 16),
                             self._reference(hits, ra, dec, 2.0, 16))


class TestFixedDirections(unittest.TestCase):
    def test_fixed_ground_direction_while_the_sky_moves(self):
        from skymon.candidates import FixedDirections
        fd = FixedDirections(hours=6)
        # the same azimuth and elevation every 10 min; the sky moves 2.5 deg in RA each time
        for k in range(3):
            n = fd.repeats(1000.0 + 600 * k, 151.0, 60.0, 190.0 + 2.5 * k, 7.0, radius_deg=1.0)
            self.assertEqual(n, k)
            fd.add(1000.0 + 600 * k, 151.0, 60.0, 190.0 + 2.5 * k, 7.0)

    def test_a_source_fixed_on_the_sky_is_not_flagged(self):
        """A pulsar repeats at one RA and Dec while its azimuth and elevation change"""
        from skymon.candidates import FixedDirections
        fd = FixedDirections(hours=6)
        for k in range(4):
            self.assertEqual(fd.repeats(1000.0 + 600 * k, 150.0 + 3 * k, 60.0 - 1.5 * k, 190.0, 7.0, 1.0), 0)
            fd.add(1000.0 + 600 * k, 150.0 + 3 * k, 60.0 - 1.5 * k, 190.0, 7.0)

    def test_repeats_before_the_sky_moved_do_not_count_and_old_ones_expire(self):
        from skymon.candidates import FixedDirections
        fd = FixedDirections(hours=1)
        fd.add(0.0, 151.0, 60.0, 190.0, 7.0)
        self.assertEqual(fd.repeats(30.0, 151.0, 60.0, 190.1, 7.0, 1.0), 0)  # same sky: no evidence
        fd.add(30.0, 151.0, 60.0, 190.1, 7.0)
        fd.add(7200.0, 10.0, 20.0, 40.0, 0.0)  # the first two expire
        self.assertEqual(fd.repeats(7300.0, 151.0, 60.0, 220.0, 7.0, 1.0), 0)
        self.assertEqual(len(fd.t), 1)


class TestPixelRing(unittest.TestCase):
    """The pixel table the monitor writes and the pixel spectra the FX imager publishes (sky_table.hpp)"""

    def test_table_layout(self):
        import mmap
        import struct

        from skymon.shm import SkyTableWriter

        name = f"skymon_test_table_{os.getpid()}"
        w = SkyTableWriter(name)
        try:
            xy_a = np.array([[10.5, 20.25], [np.nan, np.nan], [100.0, 3.5]])
            xy_b = xy_a + 1.0
            for v in (1, 2):
                self.assertEqual(w.write(256, 0.528, 1000.0, 1020.0, xy_a, xy_b), v)
            with open(w.path, "rb") as f:
                m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.assertEqual(m[0:8], b"EPICSKYT")
                ver, cap, seq, version, grid = struct.unpack_from("<IIQQI", m, 8)
                self.assertEqual((ver, cap, version, grid), (1, 3, 2, 256))
                self.assertEqual(seq % 2, 0)  # complete
                self.assertEqual(struct.unpack_from("<ddd", m, 40), (0.528, 1000.0, 1020.0))
                a = np.frombuffer(m, np.float64, 6, 128).reshape(3, 2)
                b = np.frombuffer(m, np.float64, 6, 128 + 48).reshape(3, 2)
                np.testing.assert_array_equal(a, xy_a)
                np.testing.assert_array_equal(b, xy_b)
                del a, b
                m.close()
        finally:
            w.close()
            os.unlink(w.path)

    def test_reader_takes_pixel_slots(self):
        import struct

        from skymon.shm import HEADER_BYTES, SLOT_HEADER, RingReader, RingWriter

        name = f"skymon_test_ring_{os.getpid()}"
        wr = RingWriter(name, nslots=4, max_nchan=4, max_grid=8)
        try:
            img = np.arange(4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
            wr.write(img, seq_start=1000, nseqs=1000, chan0=3072, nbin=1, grid_res=1.0, img_len_ms=41.8)
            # turn the slot into a pixel slot as the imager writes one: 4 channels x 5 pixels
            off = HEADER_BYTES  # slot 0
            pix = np.arange(20, dtype=np.float32).reshape(4, 5) + 0.5
            struct.pack_into("<IIQ", wr.mm, off + 64, 1, 5, 7)
            wr.mm[off + SLOT_HEADER:off + SLOT_HEADER + pix.nbytes] = pix.tobytes()
            rd = RingReader(name)
            meta, data = rd.read()
            self.assertEqual((meta.layout, meta.npix, meta.table_version), (1, 5, 7))
            np.testing.assert_array_equal(data, pix)
            rd.close()
        finally:
            wr.close(unlink=True)
class TestSearchEfficiency(unittest.TestCase):
    @staticmethod
    def _efficiency(nchan, nb=12, seconds=150.0):
        """Mean S/N found / ideal matched S/N for nb bursts (one per pixel, random DM and arrival) in white noise"""
        from skymon.burst import Burst, channel_fractions
        from skymon.search import Searcher

        dt = 0.041796
        e = Band(3104 - (nchan - 128) // 2, nchan).edges_mhz()
        rng = np.random.default_rng(11)
        dms, tops = rng.uniform(10, 150, nb), rng.uniform(40, seconds - 40, nb)
        P = nb + 2
        se = Searcher(e, dt, P, dm_max=300.0, block0=64, threshold=6.0, nthreads=4, warmup_images=64,
                      unix0=0.0, rng=np.random.default_rng(1), max_delay=2 * nchan)
        se.set_live(np.ones(P, bool))
        n_img = int(seconds / dt)
        sigs = [np.array([channel_fractions(Burst("b", 0, 0, d, t - K_DM * d / e[-1] ** 2, width_s=0.0), e,
                                            n * dt, dt) for n in range(n_img)]) for d, t in zip(dms, tops)]
        amps = [20.0 / np.sqrt((s ** 2).sum()) for s in sigs]  # ideal matched S/N 20
        best = np.zeros(nb)
        noise = np.random.default_rng(5)
        for n in range(n_img):
            raw = noise.standard_normal((nchan, P)).astype(np.float32) + 10.0
            for k in range(nb):
                raw[:, k] += amps[k] * sigs[k][n]
            for r in se.push(raw):
                for h in r.hits:
                    if h.slot < nb:
                        best[h.slot] = max(best[h.slot], h.snr)
            se.finish_block()
        return best / 20.0

    def test_search_efficiency_does_not_depend_on_the_channels(self):
        """192 channels (not a power of two) are searched as efficiently as 64 or 128"""
        eff = {c: self._efficiency(c) for c in (64, 128, 192)}
        for c, e in eff.items():
            self.assertTrue(np.all(e > 0), (c, e))  # every burst found
        means = {c: float(e.mean()) for c, e in eff.items()}
        self.assertGreater(means[192], 0.65, means)
        self.assertGreater(means[192], means[128] - 0.05, means)
        self.assertGreater(means[192], means[64] - 0.08, means)


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

    def test_compiled_taps_match_bicubic(self):
        """kernels.taps reproduces bicubic bit for bit, edges and free slots included"""
        from skymon.geometry import bicubic
        rng = np.random.default_rng(4)
        P, G = 5000, 128
        xy_a = rng.uniform(-2, G + 2, (P, 2))
        xy_b = xy_a + rng.normal(0, 0.3, (P, 2))
        xy_a[::50] = np.nan
        xy_b[::50] = np.nan
        edge = np.array([1.0, 2.0, 3.0, G - 2.0, G - 1.0, G - 0.5])
        xy_a[1::97, 0] = np.resize(edge, len(xy_a[1::97]))
        xy_b[1::97] = xy_a[1::97]
        xy, idx, w = np.zeros((P, 2)), np.zeros((P, 16), np.int32), np.zeros((P, 16), np.float32)
        for f in (0.0, 0.37, 1.0, 1.5):
            kernels.taps(xy_a, xy_b, f, G, xy, idx, w, nthreads=3)
            ref = xy_a + f * (xy_b - xy_a)
            ok = np.isfinite(ref[:, 0])
            i2, w2 = bicubic(np.where(ok, ref[:, 0], np.nan), np.where(ok, ref[:, 1], np.nan), G)
            np.testing.assert_array_equal(xy, ref)
            np.testing.assert_array_equal(idx, i2)
            np.testing.assert_array_equal(w, w2)

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
                     seed=5, chan0=2336, nchan=64, **cfg):
    from skymon.monitor import Config, Monitor
    from skymon.simulate import SkySimulator, tier_dt_function

    sim = SkySimulator(chan0=chan0, nchan=nchan, start_unix=T0, frb_per_hour=frb_per_hour,
                       snr=snr, dm=dm, rfi_per_hour=20, truth_path=os.path.join(out, "truth.jsonl"),
                       seed=seed, el_min=25.0,
                       tier_dt=tier_dt_function(Band(chan0, nchan).edges_mhz(), 0.083592, nchan,
                                                300.0, 64))
    mon = Monitor(Config(out_dir=out, nside=32, dm_max=300.0, block0=64, nthreads=4,
                         warmup_images=32, inject_interval_s=inject_every_s,
                         inject_snr=(25, 40), inject_dm=(10, 120), seed=seed,
                         accum_hours=0.5, **cfg))
    for _ in range(int(duration / sim.dt)):
        meta, img = sim.next_image()
        mon.process(meta, img)
    mon.store.flush()
    return sim, mon


class TestBurstFit(unittest.TestCase):
    def test_parameters_of_simulated_bursts(self):
        """
        The burst model fitted to dynamic spectra as a tier stores them (the
        injector's channel fractions, the tier's high-pass, unit noise), at Unix
        times: DM and the arrival at the band's centre within three of their
        errors, and most fits reliable.
        """
        from skymon.burstfit import _highpass, fit, fractions
        from skymon.dedisp import K_DM

        band = Band(3136, 192, 1)
        e = band.edges_mhz()
        fc = 0.5 * (e[:-1] + e[1:])
        f0 = math.sqrt(e[0] * e[-1])
        img = 8192 * 1000 / 196e6
        rng = np.random.default_rng(4)
        good = reliable = 0
        n_trials = 8
        for _ in range(n_trials):
            tier = int(rng.integers(0, 3))
            n = 2 ** tier
            dt = img * n
            dm = float(np.exp(rng.uniform(np.log(10), np.log(300))))
            w = float(np.exp(rng.uniform(np.log(0.005), np.log(0.2))))
            sweep = K_DM * dm * (e[0] ** -2 - e[-1] ** -2)
            W = int((sweep + 1.5) / dt) + 64
            t0 = 1792400000.0
            t_inf = t0 + 20 * dt - K_DM * dm * e[-1] ** -2 + rng.uniform(0, dt)
            a_hp = 1.0 / 512
            sig = fractions(e, dm, t_inf, w, 0.0, t0 + dt * np.arange(W + 1)) * (fc / f0)[:, None] ** -1.0 / math.sqrt(n)
            sig = _highpass(sig, a_hp)
            sig -= sig.mean(axis=1, keepdims=True)
            sig *= 30.0 / math.sqrt(np.sum(sig ** 2))
            noise = _highpass(rng.standard_normal(sig.shape), a_hp)
            d = sig + noise - noise.mean(axis=1, keepdims=True)
            step = 0.506 * n
            r = fit(d, t0 + dt * np.arange(W), e, dm + rng.uniform(-0.5, 0.5) * step,
                    t_inf + rng.uniform(-0.5, 0.5) * dt, dt * 2 ** round(math.log2(max(w / dt, 1))),
                    tier_images=n, a_hp=a_hp, dm_step=step)
            self.assertIsNotNone(r)
            reliable += r["reliable"]
            t_f0 = t_inf + K_DM * dm / f0 ** 2
            if r["reliable"] and abs(r["dm"] - dm) < 3 * r["dm_err"] and abs(r["t_f0"] - t_f0) < 3 * r["t_f0_err"]:
                good += 1
        self.assertGreaterEqual(reliable, n_trials - 2)
        self.assertEqual(good, reliable)


    def test_fit_in_a_worker_process_equals_the_fit_in_thread(self):
        """
        The monitor fits bursts in spawned worker processes (Monitor._submit_fit,
        burstfit.fit_job): the same parameters, to the last bit, as fit() in
        the search's thread; a failing fit comes back as an error, not an exception.
        """
        import concurrent.futures
        import multiprocessing
        from skymon.burstfit import _highpass, fit, fit_job, fractions

        band = Band(3136, 192, 1)
        e = band.edges_mhz()
        fc = 0.5 * (e[:-1] + e[1:])
        dt = 8192 * 1000 / 196e6
        t0, dm, w = 1792400000.0, 57.0, 0.02
        W = int(K_DM * dm * (e[0] ** -2 - e[-1] ** -2) / dt) + 96
        t_inf = t0 + 30 * dt - K_DM * dm * e[-1] ** -2
        a_hp = 1.0 / 512
        sig = _highpass(fractions(e, dm, t_inf, w, 0.0, t0 + dt * np.arange(W + 1)), a_hp)
        sig *= 25.0 / math.sqrt(np.sum(sig ** 2))
        d = (sig + _highpass(np.random.default_rng(2).standard_normal(sig.shape), a_hp)).astype(np.float32)
        t = t0 + dt * np.arange(W)
        args = (d, t, e, dm + 0.2, t_inf, 2 * dt, 1, a_hp, 0.506, 200)
        ref = fit(*args[:6], tier_images=1, a_hp=a_hp, dm_step=0.506, max_nfev=200)
        with concurrent.futures.ProcessPoolExecutor(
                1, mp_context=multiprocessing.get_context("spawn")) as pool:
            got = pool.submit(fit_job, *args).result(timeout=300)
            bad = pool.submit(fit_job, d[:, :4], t[:4], e, dm, t_inf, dt, 1, a_hp, 0.506, 200).result()
        self.assertIsNotNone(ref)
        self.assertEqual(got["dm"], ref["dm"])
        self.assertEqual(got["t_f0"], ref["t_f0"])
        self.assertIsNone(bad)  # too short a window: no fit, as fit() says

class TestCalibrationSteps(unittest.TestCase):
    def test_gain_changes_are_rebased_not_detected(self):
        """
        A new gain table steps every pixel's brightness by a different
        fraction (here 2%, smooth over 8 image pixels, every 20 s). Held and
        rebased, the steps leave the candidates at those of a sky without
        them; searched with the old baselines they come out as broadband
        pulses.
        """
        from skymon.monitor import Config, Monitor
        from skymon.simulate import SkySimulator, tier_dt_function

        def run(amp, rebase):
            out = tempfile.mkdtemp()
            try:
                sim = SkySimulator(chan0=2336, nchan=64, start_unix=T0, frb_per_hour=0, rfi_per_hour=0,
                                   seed=3, el_min=25.0,
                                   tier_dt=tier_dt_function(Band(2336, 64).edges_mhz(), 0.083592, 64,
                                                            300.0, 64))
                mon = Monitor(Config(out_dir=out, nside=32, dm_max=300.0, block0=64, nthreads=4,
                                     warmup_images=32, inject_interval_s=0, seed=3, accum_hours=0.5,
                                     rebase_images=rebase))
                rng = np.random.default_rng(1)
                G = sim.grid
                field = np.ones((G, G), np.float32)
                v = 0
                for i in range(int(150 / sim.dt)):
                    meta, img = sim.next_image()
                    t = i * sim.dt
                    if t >= 40 and int(t / 20) != int((t - sim.dt) / 20):
                        v += 1
                        f = np.kron(rng.standard_normal((G // 8, G // 8)), np.ones((8, 8)))
                        field = (1 + amp * f).astype(np.float32)
                    meta.gain_version = v
                    img *= field[None]
                    mon.process(meta, img)
                mon.store.flush()
                c = mon.store.counts
                return c["stored"] + int(mon.metrics.get("rfi_unrecorded", 0)), int(mon.metrics.get("rebased", 0))
            finally:
                shutil.rmtree(out, ignore_errors=True)

        quiet, _ = run(0.0, 24)
        rebased, n = run(0.02, 24)
        stepped, _ = run(0.02, 0)
        self.assertEqual(n, 6)  # changes at 40, 60, .., 140 s
        self.assertLessEqual(rebased, quiet + 2)
        self.assertGreater(stepped, quiet + 20)


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

    def test_all_channels_of_a_stream(self):
        """
        192 channels of 23.9 kHz (not a power of two), D = 2C, histories in
        half precision: bursts are found with their S/N, and the products
        hold every channel
        """
        from skymon.evaluate import report

        out = tempfile.mkdtemp()
        try:
            sim, mon = _sim_and_monitor(out, 420, frb_per_hour=40, inject_every_s=0, chan0=3072,
                                        nchan=192, delay_factor=2.0)
            se = mon.searcher
            self.assertEqual((se.C, se.D), (192, 384))
            self.assertEqual(se.tiers[0].history.dtype, np.float16)
            e = BAND192.edges_mhz()
            self.assertEqual((se.f_lo, se.f_hi), (float(e[0]), float(e[-1])))
            self.assertGreater(len(sim.truth), 0)
            rep = report(os.path.join(out, "truth.jsonl"), out, t_end=sim._t(sim.n),
                         latency_s=120)
            found = sum(c is not None for _, c in rep["rows"])
            self.assertGreaterEqual(found, max(1, int(0.7 * len(rep["truth"]))), rep["text"])
            self.assertLessEqual(len(rep["false"]), 2, rep["text"])
            # two bursts: a single burst's S/N depends on how it falls on the
            # time samples (0.56-0.84 of the target); the efficiency over many
            # is test_search_efficiency_does_not_depend_on_the_channels
            ratio = [c["snr"] / t["snr"] for t, c in rep["rows"] if c is not None]
            self.assertGreater(float(np.median(ratio)), 0.5, rep["text"])
            stored = [c for _, c in rep["rows"] if c is not None]
            with np.load(stored[0]["files"]["npz"]) as z:
                self.assertEqual(z["dynspec"].shape[0], 192)
                self.assertEqual(z["dynspec"].dtype, np.float32)
                self.assertEqual(z["dmt"].shape[0], 385)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_live_injections_are_recovered(self):
        out = tempfile.mkdtemp()
        try:
            sim, mon = _sim_and_monitor(out, 480, frb_per_hour=0, inject_every_s=40,
                                        inject_scatter=False)
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


class TestWideSearch(unittest.TestCase):
    """Per-tier boxcar widths, high-pass and trend span; the false-alarm threshold"""

    def test_search_trend_over_the_history_transform_over_the_last_samples(self):
        """kernels.search with f0: the trend over all samples, the search over the last W - f0"""
        rng = np.random.default_rng(11)
        edges = BAND.edges_mhz()
        C, D, nb, P, f0 = 64, 64, 32, 24, 40
        W = f0 + D + nb
        flat = kernels.flatten_plan(fdmt_plan(edges, D))
        t = np.arange(W, dtype=np.float32)[None, :, None]
        x = (rng.standard_normal((C, W, P)) + 0.02 * t).astype(np.float32)
        widths = np.array([1, 4, 16, 64], np.int32)
        sigma = np.full((D + 1, 4), 9.0, np.float32)
        for mode in (1, 2):
            y = x.astype(np.float64)
            y -= y.mean(axis=1, keepdims=True)
            if mode == 1:
                u = np.arange(W) - (W - 1) / 2.0
                y -= (np.tensordot(u, y, axes=(0, 1)) / np.sum(u * u))[:, None, :] * u[None, :, None]
            ref = kernels.search(np.ascontiguousarray(y[:, f0:], np.float32), flat, D, 0, nb, widths,
                                 sigma, 16, nthreads=2, detrend=False, zerodm=True)
            got = kernels.search(x, flat, D, 0, nb, widths, sigma, 16, nthreads=2, detrend=mode,
                                 zerodm=True, f0=f0)
            np.testing.assert_allclose(got[0], ref[0], rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(got[2], ref[2], rtol=1e-3, atol=1e-6 * ref[2].max())
        with self.assertRaises(ValueError):  # a boxcar longer than D + 1 samples
            kernels.search(x, flat, D, 0, nb, np.array([1, D + 2], np.int32),
                           np.ones((D + 1, 2), np.float32), 16, f0=f0)

    def test_cascade_with_a_high_pass_per_tier(self):
        from types import SimpleNamespace as NS
        rng = np.random.default_rng(12)
        C, P, W, n = 3, 11, 6, 3
        k = float(np.float32(1 / math.sqrt(2)))
        a_hp = np.array([1 / 32, 1 / 256, 1 / 64], np.float32)
        a = [NS(base=rng.standard_normal((C, P)).astype(np.float32),
                history=np.zeros((C, W, P), np.float32), pos=1,
                acc=rng.standard_normal((C, P)).astype(np.float32)) for _ in range(n)]
        b = [NS(base=t.base.copy(), history=t.history.copy(), pos=t.pos, acc=t.acc.copy())
             for t in a]
        x = rng.standard_normal((C, P)).astype(np.float32)
        kernels.tier_cascade(x, a, a_hp, k, last_dec=True, nthreads=2)
        y = x
        for j, t in enumerate(b):
            kernels.tier_push(y, t.base, t.history, t.pos, float(a_hp[j]), t.acc)
            if j + 1 < n:
                y = t.acc * np.float32(k)
                t.acc[:] = 0
        for u, v in zip(a, b):
            for f in ("base", "history", "acc"):
                np.testing.assert_array_equal(getattr(u, f), getattr(v, f))

    def test_tier_settings(self):
        from skymon.search import Searcher, parse_tier_spec, tier_spec

        self.assertEqual(parse_tier_spec("0-4:1,2,64;5-:8"),
                         [(0, 4, (1.0, 2.0, 64.0)), (5, None, (8.0,))])
        self.assertEqual(tier_spec("0-1:1,2;2-:4,8", 3), (4.0, 8.0))
        self.assertEqual(tier_spec([[1, 2], [4]], 5), (4,))
        self.assertEqual(tier_spec((32, 64), 4, True), 64)
        e = BAND.edges_mhz()
        base = Searcher(e, 0.0836, 4, dm_max=3000.0, block0=64, nthreads=1)
        wide = Searcher(e, 0.0836, 4, dm_max=3000.0, block0=64, nthreads=1,
                        widths="0-2:1,2,4,8,16,32,64;3-:1,2,4,8", detrend_factor=4,
                        far_matched=True)
        for t in base.tiers:  # unchanged by default: boxcars to 8, high-pass 32, D + block
            self.assertEqual(t.widths.tolist(), [1, 2, 4, 8])
            self.assertEqual(t.hp_tau, 32.0)
            self.assertEqual(t.history.shape[1], 64 + t.block)
            self.assertEqual(t.f0, 0)
        self.assertEqual(base.threshold, 8.0)
        for t in wide.tiers:
            wmax = 64 if t.level <= 2 else 8
            self.assertEqual(int(t.widths.max()), wmax)
            self.assertEqual(t.hp_tau, max(32.0, 4 * wmax))
            self.assertEqual(t.history.shape[1], max(64 + t.block, 4 * wmax))
            self.assertEqual(t.f0, t.history.shape[1] - 64 - t.block)
        self.assertEqual(wide.widths.tolist(), [1, 2, 4, 8, 16, 32, 64])
        # more trials, a slightly higher threshold for the same false-alarm rate
        self.assertGreater(wide.threshold, 8.0)
        self.assertLess(wide.threshold, 8.1)
        unmatched = Searcher(e, 0.0836, 4, dm_max=3000.0, block0=64, nthreads=1,
                             widths=(1, 2, 4, 8, 16, 32, 64), far_matched=False)
        self.assertEqual(unmatched.threshold, 8.0)
        small = Searcher(e, 0.0836, 4, dm_max=300.0, block0=64, nthreads=1, widths=(1, 64, 128))
        self.assertEqual(small.tiers[0].widths.tolist(), [1, 64])  # beyond D + 1 = 65: left out
        with self.assertRaises(ValueError):
            Searcher(e, 0.0836, 4, dm_max=300.0, block0=64, nthreads=1, widths=(66, 128))

    def test_far_threshold(self):
        """The applied threshold keeps the Gaussian false-alarm rate of the reference"""
        from types import SimpleNamespace as NS
        from skymon.search import far_threshold, trial_rate

        tiers = [NS(widths=np.array([1, 2, 4, 8, 16, 32, 64]), row0=0, dt=0.0418)]
        thr = far_threshold(tiers, 8.0, 256)
        q = lambda x: 0.5 * math.erfc(x / math.sqrt(2))
        r_new = trial_rate(tiers, 256)
        r_ref = trial_rate(tiers, 256, lambda t: (1, 2, 4, 8))
        self.assertAlmostEqual(r_new * q(thr), r_ref * q(8.0), delta=1e-6 * r_ref * q(8.0))
        self.assertGreater(thr, 8.0)

    def _run(self, se, P, b, amp, n_img, dt, edges, rng):
        hits = []
        for n in range(n_img):
            raw = rng.standard_normal((BAND.nchan, P)).astype(np.float32) + 10.0
            raw[:, 3] += amp * channel_fractions(b, edges, n * dt, dt)
            for res in se.push(raw):
                hits += [h for h in res.hits if h.slot == 3]
            se.finish_block()
        return max([h.snr for h in hits] or [0.0]), hits

    def test_wide_boxcars_gain_on_scattered_bursts_not_on_short_ones(self):
        from skymon.search import Searcher

        edges = BAND.edges_mhz()
        dt, P = 0.0836, 16
        dm = 25.0  # tier 0
        t_inf = 200.0 - K_DM * dm / edges[-1] ** 2
        n_img = int(330 / dt)
        cfgs = dict(base=dict(), wide=dict(widths=(1, 2, 4, 8, 16, 32, 64), detrend_factor=4))
        out = {}
        for tau1 in (0.0, 4e-5):  # 0 and about 3.7 s at the band centre
            b = Burst("x", 0, 0, dm, t_inf, width_s=0.0, tau_1ghz=tau1)
            amp = 3.0 if tau1 == 0 else 24.0
            for name, kw in cfgs.items():
                se = Searcher(edges, dt, P, dm_max=150.0, block0=128, threshold=6.0, nthreads=2,
                              warmup_images=16, unix0=0.0, rng=np.random.default_rng(3), **kw)
                se.set_live(np.ones(P, bool))
                out[(tau1, name)] = self._run(se, P, b, amp, n_img, dt, edges,
                                              np.random.default_rng(5))
        s0b, s0w = out[(0.0, "base")][0], out[(0.0, "wide")][0]
        s1b, s1w = out[(4e-5, "base")][0], out[(4e-5, "wide")][0]
        self.assertGreater(s0b, 10.0)
        self.assertAlmostEqual(s0w / s0b, 1.0, delta=0.05)  # short pulses keep their S/N
        self.assertGreater(s1w, 9.0)
        self.assertGreater(s1w, 1.5 * s1b, (s1b, s1w))  # a 3.7 s burst gains
        best = max(out[(4e-5, "wide")][1], key=lambda h: h.snr)
        self.assertGreaterEqual(best.width, 16)
        # a boxcar as wide as the sweep fixes the DM only to its width over the sweep per DM
        self.assertLess(abs(best.dm - dm), best.width_s / (K_DM * (edges[0] ** -2 - edges[-1] ** -2)))

    def test_rising_pixels_give_no_hits_with_wide_boxcars(self):
        """The monitor's default wide search (mean removal, boxcars to 128) and rising pixels"""
        from skymon.monitor import Config
        from skymon.search import Searcher

        cfg = Config()
        rng = np.random.default_rng(4)
        edges = BAND.edges_mhz()
        dt, P = 0.0836, 48
        se = Searcher(edges, dt, P, dm_max=600.0, block0=64, threshold=8.0, nthreads=2,
                      warmup_images=16, unix0=0.0, rng=rng, widths=cfg.widths,
                      detrend=cfg.detrend, stride_div=cfg.boxcar_stride_div, far_matched=True)
        se.set_live(np.ones(P, bool))
        self.assertEqual(int(se.tiers[0].widths.max()), 64)  # D + 1 = 65 here
        slope = np.linspace(0.0, 0.05 * dt, P)[None, :]
        hits = []
        for n in range(int(400 / dt)):
            raw = (rng.standard_normal((BAND.nchan, P)) + 10.0 + slope * n).astype(np.float32)
            for res in se.push(raw):
                hits += res.hits
            se.finish_block()
        self.assertLessEqual(len(hits), 1, [(h.slot, h.tier, h.width, round(h.snr, 1))
                                            for h in hits])

    def test_noise_at_fixed_widths(self):
        """The search noise against white noise at about 1, 7 and 20 s: about 1 for white noise"""
        from skymon.search import Searcher

        edges = BAND.edges_mhz()
        dt, P = 0.0836, 32
        se = Searcher(edges, dt, P, dm_max=600.0, block0=64, nthreads=2, warmup_images=16,
                      widths="0-2:1,2,4,8,16,32,64;3-:1,2,4,8", rng=np.random.default_rng(1))
        se.set_live(np.ones(P, bool))
        rng = np.random.default_rng(2)
        self.assertTrue(all(r["ratio"] is None for r in se.noise_at_widths()))
        for _ in range(int(240 / dt)):
            se.push(rng.standard_normal((BAND.nchan, P)).astype(np.float32) + 5.0)
            se.finish_block()
        got = se.noise_at_widths((1.0, 7.0, 20.0))
        self.assertEqual([r["target_s"] for r in got], [1.0, 7.0, 20.0])
        for r in got:
            self.assertLess(abs(math.log(r["width_s"] / r["target_s"])), math.log(2.0), r)
        for r in got[:2]:  # the tiers holding 1 s and 7 s have measured their noise
            self.assertIsNotNone(r["ratio"], r)
            self.assertAlmostEqual(r["ratio"], 1.0, delta=0.15, msg=r)


class TestScatteredBursts(unittest.TestCase):
    def test_exponential_scattering_conserves_fluence(self):
        edges = BAND.edges_mhz()
        dt = 0.0836
        b = Burst("x", 0, 0, 30.0, 0.0, width_s=0.001, tau_1ghz=3e-5)  # about 2.8 s at 57 MHz
        lo, hi = b.span(edges)
        ts = np.arange(lo - 1.0, hi + 1.0, dt)
        f = np.stack([channel_fractions(b, edges, t, dt) for t in ts], 1)
        np.testing.assert_allclose(f.sum(1), 1.0, atol=2e-3)
        # the mean arrival in each channel is delayed by its scattering time
        c = 0.5 * (edges[:-1] + edges[1:])
        mean_t = (f * (ts + dt / 2)[None, :]).sum(1)
        box = 0.5 * (b.arrival(edges[1:]) + b.arrival(edges[:-1]) + b.width_s)
        np.testing.assert_allclose(mean_t - box, b.tau(c), rtol=0.03)
        u = Burst("y", 0, 0, 30.0, 0.0, width_s=0.001)
        np.testing.assert_array_equal(channel_fractions(u, edges, 5.0, dt),
                                      channel_fractions(Burst("z", 0, 0, 30.0, 0.0, width_s=0.001,
                                                              tau_1ghz=0.0), edges, 5.0, dt))

    def test_injections_draw_scattering(self):
        from skymon.inject import LiveInjector

        out = tempfile.mkdtemp()
        try:
            inj = LiveInjector(out, rng=np.random.default_rng(4))
            taus = np.array([inj.draw_tau_1ghz() for _ in range(4000)])
            self.assertAlmostEqual(np.mean(taus == 0), 0.2, delta=0.03)
            t600 = taus[taus > 0] * (600.0 / 1000.0) ** -4
            self.assertGreaterEqual(t600.min(), 0.31e-3 * 0.999)
            self.assertLessEqual(t600.max(), 10e-3 * 1.001)
            # log-uniform: the median at the geometric mean of the range
            self.assertAlmostEqual(np.median(t600), math.sqrt(0.31e-3 * 10e-3), delta=3e-4)
            off = LiveInjector(out, scatter=False, rng=np.random.default_rng(4))
            self.assertEqual(off.draw_tau_1ghz(), 0.0)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_scattered_injections_are_logged(self):
        out = tempfile.mkdtemp()
        try:
            sim, mon = _sim_and_monitor(out, 420, frb_per_hour=0, inject_every_s=30,
                                        inject_tau600_ms=(0.31, 1.0),
                                        inject_unscattered_fraction=0.3,
                                        widths=(1, 2, 4, 8, 16, 32, 64))
            mon.injector.expire(math.inf)
            with open(os.path.join(out, "injections.jsonl")) as f:
                recs = [json.loads(line) for line in f]
            sched = [r for r in recs if r["event"] == "scheduled"]
            self.assertGreaterEqual(len(sched), 3)
            for r in sched:
                self.assertIn("tau_1ghz", r)
                self.assertIn("tau_s", r["meta"])
            self.assertTrue(any(r["tau_1ghz"] > 0 for r in sched), sched)
            ends = [r for r in recs if r["event"] in ("recovered", "missed")]
            self.assertTrue(ends and all("tau_1ghz" in r for r in ends))
            self.assertGreaterEqual(mon.injector.stats["recovered"], 1, mon.injector.stats)
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

    def test_dm_arms_of_a_bright_burst_merged_others_kept(self):
        """
        A bright burst also passes the threshold at trial DMs far from its own
        (the arms of its DM-time plane). Those are merged into it, whichever
        comes first; a faint burst at another DM elsewhere, a bright one at the
        same place long after, and a detection stronger than an arm can be are kept.
        """
        from skymon.candidates import CandidateStore, dm_arm_snr

        out = tempfile.mkdtemp()
        try:
            store = CandidateStore(out)
            beam, t0 = 3.0, 1792008000.0
            # arms seen first, at DMs 40 and 160 (S/N about 25 and 20), then the burst, DM 80 S/N 250
            self.assertIsNotNone(store.add(self.cand("arm40", 25.0, 200.0, 34.0, t0 - 3.0, dm=40.0), None, beam))
            self.assertIsNotNone(store.add(self.cand("arm160", 20.0, 200.5, 34.0, t0 + 6.0, dm=160.0), None, beam))
            self.assertIsNotNone(store.add(self.cand("burst", 250.0, 200.0, 34.0, t0, dm=80.0), None, beam))
            # an arm seen after the burst
            self.assertIsNone(store.add(self.cand("arm10", 15.0, 199.5, 34.0, t0 - 8.0, dm=10.0), None, beam))
            # kept: another burst 8 deg away; one at the same place 10 min later;
            # and one too strong for an arm (S/N 120 at 40 DM units off)
            self.assertGreater(120.0, 2 * dm_arm_snr(250.0, 0.17, 40.0, 55.9, 58.9))
            self.assertIsNotNone(store.add(self.cand("far", 20.0, 208.0, 34.0, t0, dm=40.0), None, beam))
            self.assertIsNotNone(store.add(self.cand("later", 30.0, 200.0, 34.0, t0 + 600.0, dm=40.0), None, beam))
            self.assertIsNotNone(store.add(self.cand("strong", 120.0, 200.0, 34.0, t0 - 3.0, dm=40.0), None, beam))
            store.flush()
            self.assertEqual(sorted(c.id for c in store.list()), ["burst", "far", "later", "strong"])
            burst = store.get("burst")
            self.assertEqual(sorted(a["id"] for a in burst.dm_arms), ["arm10", "arm160", "arm40"])
            self.assertEqual(store.counts["dm_arms"], 3)
            # the arms stored before the burst keep their files, marked as superseded by it
            with open(os.path.join(out, "candidates", time.strftime("%Y%m%d", time.gmtime(t0 - 33.0)),
                                   "arm40.json")) as f:
                self.assertEqual(json.load(f)["files"]["superseded_by"], "burst")
        finally:
            shutil.rmtree(out, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
