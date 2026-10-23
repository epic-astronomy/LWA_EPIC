"""
The long-accumulation writer in a process of its own (src/ex/fits_writer_proc.hpp,
src/python/epic_fits_writer.py): the files it writes equal those of the
in-process writer, the metadata keep their types, and a writer that dies is
replaced at the next job.

Runs a small C++ client (tests/cpp/fits_writer_proc_test.cpp, built with g++).
The daily digest is pointed at an unreachable database, so nothing is written
to one.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = os.path.join(ROOT, "src", "python")
sys.path.insert(0, PY)

# no daily digest rows anywhere: the digest fails to connect, quickly
os.environ["EPIC_NODE_DSN"] = "host=127.0.0.1 port=1 dbname=none user=none connect_timeout=1"


def image(n, nchan=2, grid=8):
    c, a, b, k = np.meshgrid(np.arange(nchan), np.arange(grid), np.arange(grid), np.arange(4), indexing="ij")
    return (n * 1000.0 + c * 100.0 + a * 10.0 + b + 0.1 * k).astype(np.float32).ravel()


def meta(n):
    return {"time_tag": 392536608000000000 + n * 196000000 * 300, "img_len_ms": 300000.0, "nimg": 7178,
            "grid_size": 8, "grid_res": 1.0, "chan0": 3136, "nchan": 2, "chan_width": 1, "cfreq": 77.304,
            "gain_version": 9, "support": 5, "note": 'a "quoted" note'}


class TestFitsWriterProcess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gxx = shutil.which("g++")
        if not gxx:
            raise unittest.SkipTest("g++ not available")
        cls.tmp = tempfile.mkdtemp()
        cls.exe = os.path.join(cls.tmp, "fwp")
        subprocess.run([gxx, "-std=c++17", "-O1", "-o", cls.exe,
                        os.path.join(ROOT, "tests", "cpp", "fits_writer_proc_test.cpp")], check=True)
        cls.env = dict(os.environ, EPIC_PYTHON=sys.executable,
                       PYTHONPATH=PY + os.pathsep + os.environ.get("PYTHONPATH", ""))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_client(self, out, njobs, *extra):
        r = subprocess.run([self.exe, out, str(njobs), *extra], env=self.env, capture_output=True,
                           text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.splitlines(), r.stderr

    def test_files_equal_the_in_process_writer(self):
        from astropy.io import fits
        import epic_utils

        out_p = os.path.join(self.tmp, "proc")
        out_i = os.path.join(self.tmp, "inproc")
        lines, _ = self.run_client(out_p, 2)
        self.assertEqual(len(lines), 2)
        for n, path in enumerate(lines):
            self.assertFalse(path.startswith("ERROR"), path)
            self.assertTrue(os.path.isfile(path), path)
            self.assertTrue(path.startswith(out_p + "/"), path)
            ref = epic_utils.save_output(image(n), 8, 2, out_i + "/", meta(n))
            self.assertEqual(os.path.basename(ref), os.path.basename(path))
            with fits.open(path) as a, fits.open(ref) as b:
                for ha, hb in zip(a, b):
                    np.testing.assert_array_equal(ha.data, hb.data)
                    self.assertEqual(dict(ha.header), dict(hb.header))
                # integers stay integers through the JSON header
                self.assertIsInstance(a[0].header["TIME_TAG"], int)
                self.assertEqual(a[0].header["TIME_TAG"], meta(n)["time_tag"])
                self.assertIsInstance(a[0].header["GRID_RES"], float)
                self.assertEqual(a[0].header["NOTE"], 'a "quoted" note')
            self.assertTrue(os.path.isfile(path[:-len(".fits")] + "_QL.png"))

    def test_a_dead_writer_is_replaced(self):
        out = os.path.join(self.tmp, "kill")
        lines, err = self.run_client(out, 3, "kill")
        self.assertEqual(len(lines), 3)
        self.assertTrue(os.path.isfile(lines[0]), lines)
        # the job after the kill finds the old writer gone and starts a new one;
        # at most that job is lost, the next is written
        self.assertTrue(os.path.isfile(lines[2]), (lines, err))

    def test_unwritable_directory_reports_and_keeps_the_writer(self):
        lines, _ = self.run_client("/proc/no_such_dir", 2)
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(l.startswith("ERROR") for l in lines), lines)


if __name__ == "__main__":
    unittest.main()
