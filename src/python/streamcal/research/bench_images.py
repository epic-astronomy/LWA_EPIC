"""Compare GPU images dumped by `epic++ --bench --bench_dump` at different gain versions.

    python bench_images.py DUMPDIR ORACLEDIR [--chans 0:128]

Images are float32 [chan][x][y][4] (XX, YY, Re XY*, Im XY*), unshifted. For each gain
version, the XX and YY images averaged over the dumps and channels are compared with
the oracle (exact gains): normalised residual after fitting a scale, and the peak at
the brightest oracle pixel relative to the oracle.
"""
import argparse
import collections
import glob
import os
import re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dumps")
ap.add_argument("oracle")
ap.add_argument("--grid", type=int, default=128)
ap.add_argument("--nchan", type=int, default=128)
a = ap.parse_args()


def load(fn):
    x = np.fromfile(fn, np.float32).reshape(a.nchan, a.grid, a.grid, 4)
    return x[..., 0].mean(0) + x[..., 1].mean(0)  # channel-averaged XX + YY


def groups(d):
    g = collections.defaultdict(list)
    for fn in sorted(glob.glob(os.path.join(d, "img_*.f32"))):
        m = re.search(r"seq(\d+)_gv(\d+)", fn)
        g[int(m.group(2))].append((int(m.group(1)), fn))
    return g


og = groups(a.oracle)
ref = np.mean([load(fn) for v in og for _, fn in og[v]], axis=0)
ref = np.fft.fftshift(ref)
pk = np.unravel_index(np.argmax(ref), ref.shape)
yy, xx = np.mgrid[:a.grid, :a.grid] - a.grid / 2
sky = np.hypot(xx, yy) < 0.45 * a.grid  # inside the horizon
print(f"oracle: {sum(len(v) for v in og.values())} images; brightest pixel {pk}")
for v, items in sorted(groups(a.dumps).items()):
    img = np.fft.fftshift(np.mean([load(fn) for _, fn in items], axis=0))
    sc = np.sum(img[sky] * ref[sky]) / np.sum(img[sky] ** 2)
    res = np.linalg.norm(sc * img[sky] - ref[sky]) / np.linalg.norm(ref[sky])
    print(f"gain v{v:4d}: {len(items):3d} images, residual vs oracle {res:.4f}, peak ratio {img[pk] * sc / ref[pk]:.3f}, "
          f"first seq {items[0][0]}")
