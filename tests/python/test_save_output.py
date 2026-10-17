"""
The accumulated FITS image of epic_utils.save_output: its times and celestial
WCS refer to the middle of the accumulation.

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
        epic_utils.ingest_daily_digest = lambda ih, ph, data, cfreq: seen.update(ih=ih)
        d = tempfile.mkdtemp(prefix="save-output-")
        try:
            grid, nchan, img_len_ms = 64, 2, 600000.0  # a 10 min accumulation
            t_beg = 1792252800.0
            meta = dict(time_tag=int(t_beg * epic_utils.FS), img_len_ms=img_len_ms,
                        grid_size=grid, grid_res=1.056, chan0=2400, nchan=nchan)
            arr = np.zeros(nchan * grid * grid * 4, dtype=np.float32)
            path = epic_utils.save_output(arr, grid, nchan, d + "/", meta)
            t_mid = t_beg + 300.0
            self.assertEqual(os.path.basename(path), f"EPIC_{t_mid:.3f}_{(2400 + 0) * epic_utils.CHAN_BW / 1e6:.3f}MHz.fits")
            with fits.open(path) as h:
                ph, ih = h[0].header, h[1].header
            for key, t in (("DATE-OBS", t_mid), ("DATE-BEG", t_beg), ("DATE-END", t_beg + 600.0)):
                self.assertAlmostEqual(Time(ph[key]).unix, t, places=3, msg=key)
            self.assertAlmostEqual(Time(ih["DATETIME"]).unix, t_mid, places=3)
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


if __name__ == "__main__":
    unittest.main()
