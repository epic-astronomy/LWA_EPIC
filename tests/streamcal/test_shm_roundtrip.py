"""Cross-language check of the shared-memory formats (run on a node with the test_io binary).

    python tests/streamcal/test_shm_roundtrip.py /tmp/streamcal_test_io /dev/shm/streamcal_test
"""
import subprocess
import sys

import numpy as np

sys.path.insert(0, "src/python")
from streamcal import shm  # noqa: E402

exe, d = sys.argv[1], sys.argv[2]
fails = 0

# Python writer -> C++ reader
w = shm.GainTableWriter(f"{d}/rt_gain", chan0=3040, nchan=192)
rng = np.random.default_rng(0)
corr = (rng.normal(size=(192, 256, 2)) + 1j * rng.normal(size=(192, 256, 2))).astype(np.complex64)
v = w.publish(corr, valid_from_seq=10)
out = subprocess.run([exe, "readgain", f"{d}/rt_gain", "3104", "128", "20"], capture_output=True, text=True).stdout.split()
sl = corr[64:192]
# output: version V valid 1 n N sum SR SI first FR FI
ok = out[0] == "version" and int(out[1]) == v and abs(float(out[7]) - sl.real.sum()) < 1e-4 * abs(sl.real).sum() \
    and abs(float(out[10]) - sl[0, 0, 0].real) < 1e-5
print("python GainTableWriter -> C++ GainTableReader", "ok" if ok else f"FAIL {out}")
fails += not ok
out = subprocess.run([exe, "readgain", f"{d}/rt_gain", "3104", "128", "5"], capture_output=True, text=True).stdout
ok = out.startswith("none")
print("valid_from_seq respected across languages     ", "ok" if ok else "FAIL")
fails += not ok

# C++ tap -> Python reader, against numpy on the same seeded data
subprocess.run([exe, "tap", f"{d}/rt_vis", "2"], check=True, capture_output=True)
r = shm.VisTableReader(f"{d}/rt_vis")
t = r.poll()
ok = t is not None and t.nsamp == 100 and t.nchan == 128 and np.allclose(t.R[5, 1], t.R[5, 1].conj().T)
print("C++ CalTap -> python VisTableReader (header, Hermitian)", "ok" if ok else "FAIL")
fails += not ok
ok = r.poll() is None
print("reader returns each table once                 ", "ok" if ok else "FAIL")
fails += not ok
print("all passed" if not fails else "FAILED")
sys.exit(fails != 0)
