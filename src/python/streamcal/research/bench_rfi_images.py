"""Image noise and source S/N with and without the interference projection (epic++ --bench dumps).

    python bench_rfi_images.py DUMPDIR [DUMPDIR ...] [--min-gv 2]

Images are float32 [chan][x][y][4] (XX, YY, Re XY*, Im XY*), unshifted; Stokes I = XX + YY
averaged over channels. Per directory and projector state (pv = 0: none), using images with
gain version >= --min-gv (calibrated):
  noise: robust rms over the sky of the difference of consecutive dumps / sqrt(2) (independent
         data, the same sky), median over pairs;
  peak:  brightest pixel of the mean image above the sky median, and its S/N against that noise.
"""
import argparse
import collections
import glob
import os
import re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--grid", type=int, default=128)
ap.add_argument("--nchan", type=int, default=128)
ap.add_argument("--min-gv", type=int, default=2)
a = ap.parse_args()
yy, xx = np.mgrid[:a.grid, :a.grid] - a.grid / 2
sky = np.hypot(xx, yy) < 0.45 * a.grid


def load(fn):
    x = np.fromfile(fn, np.float32).reshape(a.nchan, a.grid, a.grid, 4)
    return np.fft.fftshift(x[..., 0].mean(0) + x[..., 1].mean(0))


def rms(v):
    return 1.4826 * np.median(np.abs(v - np.median(v)))


for d in a.dirs:
    g = collections.defaultdict(list)
    for fn in sorted(glob.glob(os.path.join(d, "img_*.f32"))):
        m = re.search(r"seq(\d+)_gv(\d+)_pv(\d+)", fn)
        if m and int(m.group(2)) >= a.min_gv:
            g["projected" if int(m.group(3)) >= 2 else ("cold projector" if int(m.group(3)) == 1 else "raw")].append(
                (int(m.group(1)), fn))
    for state, items in sorted(g.items()):
        items.sort()
        imgs = [load(fn) for _, fn in items]
        if len(imgs) < 3:
            continue
        noise = np.median([rms((imgs[i + 1] - imgs[i])[sky]) / np.sqrt(2) for i in range(len(imgs) - 1)])
        mean = np.mean(imgs, axis=0)
        base = np.median(mean[sky])
        pk = np.unravel_index(np.argmax(np.where(sky, mean, -np.inf)), mean.shape)
        print(f"{os.path.basename(os.path.dirname(d.rstrip('/')))}/{state}: {len(imgs)} images; noise {noise:.4g}; "
              f"peak {mean[pk] - base:.4g} at {pk}, S/N {(mean[pk] - base) / noise:.1f}; "
              f"sky median {base:.4g}", flush=True)
