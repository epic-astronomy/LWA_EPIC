"""
The accumulated FITS image of epic_utils.save_output: its times and celestial
WCS refer to the middle of the accumulation; it goes to a daily directory;
partial sums carry their image count; the digest is a per-image mean; and a
directory that cannot be written drops the accumulation without raising.

    cd tests/python && python -m unittest -v test_save_output
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src", "python"))


class TestSaveOutput(unittest.TestCase):
    def test_times_and_wcs_at_mid_accumulation(self):
        from astropy.utils import iers
        iers.conf.auto_download = False  # the workers have no internet access
        iers.conf.iers_degraded_accuracy = "warn"
        import epic_utils
        from astropy.io import fits
        from astropy.time import Time
        from epic_geometry import apply_wcs

        seen = {}
        orig = epic_utils.ingest_daily_digest
        epic_utils.ingest_daily_digest = lambda ih, ph, data, cfreq, nimg=1: seen.update(ih=ih, nimg=nimg)
        d = tempfile.mkdtemp(prefix="save-output-")
        try:
            grid, nchan, img_len_ms = 64, 2, 600000.0  # a 10 min accumulation
            t_beg = 1792252800.0
            meta = dict(time_tag=int(t_beg * epic_utils.FS), img_len_ms=img_len_ms,
                        grid_size=grid, grid_res=1.056, chan0=2400, nchan=nchan, nimg=7178)
            arr = np.zeros(nchan * grid * grid * 4, dtype=np.float32)
            path = epic_utils.save_output(arr, grid, nchan, d + "/", meta)
            t_mid = t_beg + 300.0
            self.assertEqual(path, os.path.join(d, "20261017", f"EPIC_{t_mid:.3f}_"
                                                f"{2400 * epic_utils.CHAN_BW / 1e6:.3f}MHz.fits"))
            self.assertTrue(os.path.exists(path.replace(".fits", "_QL.png")))
            self.assertFalse(os.path.exists(path + ".part"))
            self.assertEqual(seen["nimg"], 7178)
            with fits.open(path) as h:
                ph, ih = h[0].header, h[1].header
            for key, t in (("DATE-OBS", t_mid), ("DATE-BEG", t_beg), ("DATE-END", t_beg + 600.0)):
                self.assertAlmostEqual(Time(ph[key]).unix, t, places=3, msg=key)
            self.assertAlmostEqual(Time(ih["DATETIME"]).unix, t_mid, places=3)
            self.assertEqual(ph["NIMGACC"], 7178)
            self.assertEqual(seen["ih"]["DATETIME"], ih["DATETIME"])  # the digest's time
            ref = fits.Header()
            apply_wcs(ref, Time(t_mid, format="unix"), grid, 1.056)
            self.assertAlmostEqual(ih["CRVAL1"], ref["CRVAL1"], places=6)
            self.assertAlmostEqual(ih["CRVAL2"], ref["CRVAL2"], places=6)
            beg = fits.Header()
            apply_wcs(beg, Time(t_beg, format="unix"), grid, 1.056)
            self.assertGreater(abs(ih["CRVAL1"] - beg["CRVAL1"]), 1.0)  # 5 min of drift: 1.25 deg
        finally:
            epic_utils.ingest_daily_digest = orig
            shutil.rmtree(d, ignore_errors=True)


    def _meta(self, nimg, t_beg=1792252800.0):
        import epic_utils
        return dict(time_tag=int(t_beg * epic_utils.FS), img_len_ms=83.592 * nimg,
                    grid_size=32, grid_res=1.056, chan0=2400, nchan=1, nimg=nimg)

    def test_unwritable_directory_drops_the_accumulation(self):
        from astropy.utils import iers
        iers.conf.auto_download = False
        iers.conf.iers_degraded_accuracy = "warn"
        import epic_utils
        d = tempfile.mkdtemp(prefix="save-output-")
        orig = epic_utils.ingest_daily_digest
        epic_utils.ingest_daily_digest = lambda *a, **k: None
        try:
            arr = np.zeros(32 * 32 * 4, dtype=np.float32)
            blocker = os.path.join(d, "not-a-dir")
            with open(blocker, "w") as f:
                f.write("x")
            self.assertEqual(epic_utils.save_output(arr, 32, 1, blocker, self._meta(10)), "")
            ro = os.path.join(d, "ro")
            os.makedirs(ro)
            os.chmod(ro, 0o555)
            if os.access(ro, os.W_OK):
                self.skipTest("running as root")
            self.assertEqual(epic_utils.save_output(arr, 32, 1, ro, self._meta(10)), "")
            self.assertEqual(os.listdir(ro), [])
        finally:
            epic_utils.ingest_daily_digest = orig
            os.chmod(os.path.join(d, "ro"), 0o755)
            shutil.rmtree(d, ignore_errors=True)

    def test_digest_is_a_per_image_mean(self):
        import daily_digest as dd
        rows = []

        class Conn:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, stmt):
                rows.extend(stmt.compile().params.values())

            def commit(self):
                pass

        class Engine:
            def connect(self):
                return Conn()

        orig = dd.get_pixel_indices2, dd._get_engine
        dd.get_pixel_indices2 = lambda ih, ph, el: dict(
            nsrc=np.array([1]), ncoords=np.array([4]), src_ids=["sun"] * 4,
            pix_x=np.array([1, 2, 1, 2]), pix_y=np.array([1, 1, 2, 2]))
        dd._get_engine = lambda: Engine()
        try:
            one = np.ones((4, 3, 8, 8), dtype=np.float32)  # one image: XX = YY = 1
            out = {}
            for nimg in (100, 40):  # a full and a partial sum of the same sky
                rows.clear()
                dd.ingest_daily_digest({"DATETIME": "2026-10-17T12:00:00"}, {}, one * nimg, 57e6,
                                       nimg=nimg)
                out[nimg] = [v for v in rows if isinstance(v, list) and len(v) == 3][0]
            self.assertEqual(out[100], out[40])
            self.assertEqual(out[100], [8.0, 8.0, 8.0])  # 2 pols x 4 pixels, per image
        finally:
            dd.get_pixel_indices2, dd._get_engine = orig


if __name__ == "__main__":
    unittest.main()
