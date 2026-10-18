"""Follow the gain table published by streamcal and score it against the benchmark truth.

    python bench_watch.py MODELDIR /dev/shm/epic_gains_gpu0 [--oracle]

The corrections c multiply epic++'s nominal phases; with the true residual gains eps
of the dataset, c * eps must be constant across stands. Prints, per new table: phase
error (tip/tilt removed) and equivalent image shift, amplitude error, flags on the dead
inputs. --oracle writes the exact table 1/eps instead (reference images).
"""
import argparse
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402

from streamcal import geometry as G, shm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("gains")
ap.add_argument("--oracle", action="store_true")
ap.add_argument("--duration", type=float, default=1e9)
a = ap.parse_args()
z = np.load(os.path.join(a.model, "truth.npz"))
eps, chans, dead = z["eps"], z["chans"], z["dead"]  # eps [pol, 256, nchan]
used = np.flatnonzero(G.eg.used_stands())
live = np.setdiff1d(used, dead)
r = G.eg.stand_xyz()
freq = chans * G.CHAN_BW
if a.oracle:
    corr = np.zeros((len(chans), 256, 2), np.complex64)
    for p in range(2):
        c = np.zeros((256, len(chans)), complex)
        c[live] = 1 / eps[p, live]
        corr[:, :, p] = c.T
    w = shm.GainTableWriter(a.gains, int(chans[0]), len(chans), solver="oracle")
    print("oracle table version", w.publish(corr))
    sys.exit(0)


def score(corr):
    out = []
    for p in range(2):
        q = corr[:, live, p] * eps[p, live].T  # [chan, stand]
        e_ph, e_sh, e_amp = [], [], []
        for ci in range(len(chans)):
            x = q[ci] / np.mean(q[ci])
            ph = np.angle(x)
            A = np.c_[np.ones(len(ph)), 2 * np.pi * freq[ci] / G.SOL * r[live, :2]]
            cfit = np.linalg.lstsq(A, ph, rcond=None)[0]
            e_ph.append(np.degrees(np.sqrt(np.mean(np.angle(x * np.exp(-1j * (A @ cfit))) ** 2))))
            e_sh.append(np.degrees(np.hypot(*cfit[1:])))
            e_amp.append(100 * np.std(np.abs(x)) / np.mean(np.abs(x)))
        out.append((np.median(e_ph), np.median(e_sh), np.median(e_amp), bool(np.all(corr[:, dead, p] == 0))))
    return out


last = -1
t0 = time.time()
while time.time() - t0 < a.duration:
    try:
        raw = open(a.gains, "rb").read()
        h = shm.GAIN_HDR.unpack_from(raw, 0)
    except (FileNotFoundError, struct.error):
        time.sleep(1)
        continue
    if h[0] == b"EPICGAIN" and h[4] != last and not h[3] & 1:
        last = h[4]
        corr = np.frombuffer(raw[128:], np.complex64).reshape(h[6], 256, 2)
        s = score(corr)
        print(f"{time.strftime('%H:%M:%S')} gains v{h[4]}: X phase {s[0][0]:.2f} deg (shift {s[0][1]:.3f} deg), amp {s[0][2]:.1f}%, "
              f"dead flagged {s[0][3]} | Y phase {s[1][0]:.2f} deg (shift {s[1][1]:.3f} deg), amp {s[1][2]:.1f}%, "
              f"dead flagged {s[1][3]}", flush=True)
    time.sleep(1)
