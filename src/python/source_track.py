"""
Exact pixel track of a source on the EPIC image, for analyses run outside
the imager.

    python source_track.py --ra 188.2 --dec 63.16 --t0 1792008000 --t1 1792011600
        [--step 10] [--grid-size 128] [--grid-res 1.056] [--body sun]

prints JSON {"t": [...unix s], "x": [...], "y": [...]} with the fractional,
1-based FITS pixel position of the source at each time, computed with the
same celestial WCS the imager uses when it extracts pixels
(pixel_extractor.get_image_headers_sv and epic_geometry.apply_wcs). Between
the sampled times the position changes smoothly and can be interpolated.
"""

import argparse
import json

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from epic_geometry import apply_wcs
from pixel_extract_utils import DynSources


def track(ra, dec, t0, t1, step=10.0, grid_size=128, grid_res=1.056, body=None):
    times = np.arange(t0, t1 + step, step)
    xs, ys = [], []
    for t in times:
        obstime = Time(t, format="unix", precision=6)
        header = fits.ImageHDU(np.zeros((grid_size, grid_size))).header
        apply_wcs(header, obstime, grid_size, grid_res)
        if body:
            ra, dec = DynSources.get_lwasv_skypos(body, obstime.isot)
        x, y = WCS(header, naxis=2).all_world2pix(ra, dec, 1, ra_dec_order=True)
        xs.append(float(x))
        ys.append(float(y))
    return dict(t=[float(t) for t in times], x=xs, y=ys)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ra", type=float, default=0.0)
    p.add_argument("--dec", type=float, default=0.0)
    p.add_argument("--body", default=None, help="solar-system body (e.g. sun) instead of ra/dec")
    p.add_argument("--t0", type=float, required=True)
    p.add_argument("--t1", type=float, required=True)
    p.add_argument("--step", type=float, default=10.0)
    p.add_argument("--grid-size", type=int, default=128)
    p.add_argument("--grid-res", type=float, default=1.056)
    a = p.parse_args()
    print(json.dumps(track(a.ra, a.dec, a.t0, a.t1, a.step, a.grid_size, a.grid_res, a.body)))
