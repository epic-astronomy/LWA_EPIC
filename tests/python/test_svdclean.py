"""
The Marchenko-Pastur cut (skymon/svdclean.py): a flash shared by many pixels is
removed, a faint burst in one pixel is kept, and the monitor still recovers its
injections and handles calibration changes with the block buffer in front of
its search.
"""
import math
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "python"))

from test_skymon import T0, _sim_and_monitor  # noqa: E402


class TestCleanBlock(unittest.TestCase):
    def test_flash_removed_and_faint_burst_kept(self):
        from skymon.svdclean import SvdCleaner, clean_block

        rng = np.random.default_rng(2)
        W, C, P = 256, 16, 3000
        mu = np.full((C, P), 100.0, np.float32)
        var = np.full((C, P), 4.0, np.float32)
        raw = (100.0 + 2.0 * rng.standard_normal((W, C, P))).astype(np.float32)
        # flashes through the sidelobes: a fixed pattern over 40% of the sky, 3 sigma, 6 images
        pattern = np.where(rng.random(P) < 0.4, rng.standard_normal(P), 0).astype(np.float32)
        flash_t = [30, 31, 90, 150, 151, 200]
        for t in flash_t:
            raw[t] += 2.0 * 3.0 * pattern[None, :]
        # a faint burst: one pixel, one image, 1.5 sigma in each of 16 channels (S/N 6)
        bp, bt = 1234, 120
        raw[bt, :, bp] += 2.0 * 1.5
        before_flash = np.abs((raw[flash_t].mean(1) - 100.0) / 2.0 * np.sqrt(C))[:, pattern != 0].mean()
        before_burst = ((raw[bt, :, bp] - 100.0) / 2.0).sum() / np.sqrt(C)
        cl = SvdCleaner(C, nsub=4, window=W)
        cl.clean(raw, mu, var)
        after_flash = np.abs((raw[flash_t].mean(1) - 100.0) / 2.0 * np.sqrt(C))[:, pattern != 0].mean()
        after_burst = ((raw[bt, :, bp] - 100.0) / 2.0).sum() / np.sqrt(C)
        self.assertGreater(min(cl.status()["k"]), 0)
        self.assertLess(after_flash, 0.2 * before_flash, (before_flash, after_flash))
        self.assertGreater(after_burst, 0.9 * before_burst, (before_burst, after_burst))
        self.assertEqual(cl.status()["images"], W)

    def test_pure_noise_is_left_alone(self):
        from skymon.svdclean import clean_block

        rng = np.random.default_rng(4)
        W, C, P = 256, 8, 2000
        mu = np.zeros((C, P), np.float32)
        var = np.ones((C, P), np.float32)
        raw = rng.standard_normal((W, C, P)).astype(np.float32)
        ref = raw.copy()
        info = clean_block(raw, mu, var, [(0, 4), (4, 8)])
        # at most a few components at the edge; the noise barely changes
        self.assertLessEqual(max(i["k"] for i in info), 3)
        self.assertLess(float(np.std(raw - ref)), 0.15)


class TestMonitorWithCleaning(unittest.TestCase):
    def test_injections_recovered_with_the_block_buffer(self):
        out = tempfile.mkdtemp()
        try:
            sim, mon = _sim_and_monitor(out, 480, frb_per_hour=0, inject_every_s=40,
                                        inject_scatter=False, svd_clean=True, svd_window=128)
            mon.injector.expire(math.inf)
            s = mon.injector.stats
            st = mon.svd.status()
            self.assertGreater(st["blocks"], 10)
            self.assertGreaterEqual(s["injected"], 3)
            self.assertGreaterEqual(s["recovered"], 0.7 * s["injected"], s)
            # every image was searched: the buffer holds at most one block
            self.assertLessEqual(len(mon._svd_buf), 128)
        finally:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
