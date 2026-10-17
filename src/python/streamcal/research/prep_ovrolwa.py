"""Build the OVRO-LWA hybrid diffuse map used by sky.diffuse(model="ovrolwa").

    python prep_ovrolwa.py MAPDIR [OUT.npz] [CONTRAST]

MAPDIR holds the LAMBDA release of the OVRO-LWA m-mode maps (Eastwood et al. 2018,
AJ 156, 32): ovro_lwa_sky_map_{36.528,41.760,46.992,52.224}MHz.fits, HEALPix nside 2048,
RING, Galactic, brightness temperature in K, 15-26 arcmin resolution. They are read
as FITS arrays only.

Properties of the release that matter here, measured against LFSM:
  * pixels south of dec -30 deg (and a 2 deg disc around the Sun's position during the
    observation) are exactly 0;
  * the monopole and the structure that depends on declination alone (m = 0 in the
    equatorial frame) are missing: m-mode analysis without autocorrelations does not
    constrain them, so the map runs about -14 kK below LFSM at 36.5 MHz, with an offset
    that varies with declination;
  * the eight brightest northern sources (Cyg A, Cas A, Tau A, Vir A, Her A, Hya A,
    3C 123, 3C 353) are already removed, leaving residuals of about 1-2 kK within 0.5 deg
    (about 1% of the Cyg A flux);
  * away from the missing modes the map agrees with LFSM to within its 10% scale
    (OVRO-LWA = 1.10 x LFSM + offset, correlation 0.97 at dec < 15 deg).

The hybrid map at each OVRO-LWA frequency:
  T = T_ovro + [ring_mean_LFSM(dec) - ring_mean_OVRO(dec)]   for dec >= -30 deg,
  T = T_LFSM                                                  south of -30 deg and in the Sun hole,
with a linear blend over dec -30..-26 deg (the ring means are those of the A-team-inpainted
LFSM, computed on 0.5 deg declination rings and smoothed over 1.5 deg). The four A-team
positions are then inpainted within 1 deg (residuals of the OVRO-LWA source removal); the
Cygnus X complex 2-3 deg from Cyg A is kept. Output: nside 256 maps in K (float32), the
frequencies, and the OVRO-LWA coverage mask.

CONTRAST (default 1) scales the OVRO-LWA structure about the LFSM ring means,
T = ring_mean_LFSM + CONTRAST (T_ovro - ring_mean_OVRO), for testing the flux scale of the
small-scale structure (OVRO-LWA = 1.07-1.17 x LFSM on the common sky).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import healpy as hp  # noqa: E402
import astropy.units as u  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402

from streamcal import sky as S  # noqa: E402

NODES = (36.528, 41.760, 46.992, 52.224)
NS_WORK = 512
NS_OUT = 256
DEC_MIN = -30.0
BLEND = 4.0  # deg


def main(mapdir, out, contrast=1.0):
    npix = hp.nside2npix(NS_WORK)
    th, ph = hp.pix2ang(NS_WORK, np.arange(npix))
    dec = SkyCoord(l=ph * u.rad, b=(np.pi / 2 - th) * u.rad, frame="galactic").icrs.dec.deg
    edges = np.arange(-90, 90.5, 0.5)
    ring = np.clip(np.digitize(dec, edges) - 1, 0, len(edges) - 2)
    cen = 0.5 * (edges[1:] + edges[:-1])
    lf_maps = S._lfsm_native(NODES)  # [4, npix 256], K, A-team inpainted
    maps, cover = [], None
    for nu, lf in zip(NODES, lf_maps):
        fn = os.path.join(mapdir, f"ovro_lwa_sky_map_{nu:.3f}MHz.fits")
        m = hp.read_map(fn, dtype=np.float64)
        obs = np.isfinite(m) & (m != 0) & (m != hp.UNSEEN)
        mw = hp.ud_grade(np.where(obs, m, hp.UNSEEN), NS_WORK)
        ow = mw != hp.UNSEEN
        lw = hp.ud_grade(lf, NS_WORK)
        ok = ow & (dec >= DEC_MIN)
        n = np.bincount(ring[ok], minlength=len(cen))
        mo = np.bincount(ring[ok], mw[ok], minlength=len(cen)) / np.maximum(n, 1)
        ml = np.bincount(ring[ok], lw[ok], minlength=len(cen)) / np.maximum(n, 1)
        good = n > 20
        mo, ml = (np.convolve(np.interp(cen, cen[good], x[good]), np.ones(3) / 3, mode="same") for x in (mo, ml))
        d = ml - mo
        h = np.where(ow, np.interp(dec, cen, ml) + contrast * (mw - np.interp(dec, cen, mo)), lw)
        w = np.clip((dec - DEC_MIN) / BLEND, 0, 1) * ow
        h = w * h + (1 - w) * lw
        h = S.remove_compact(h, r_in_deg=1.0, r_out_deg=2.0)
        o = hp.ud_grade(h, NS_OUT)
        maps.append(o.astype(np.float32))
        cov = hp.ud_grade(w, NS_OUT)
        cover = cov if cover is None else np.minimum(cover, cov)
        # diagnostics against LFSM on the jointly covered sky
        lo = hp.ud_grade(lw, NS_OUT)
        s = cov > 0.99
        a, b = np.polyfit(lo[s], o[s], 1)
        sm = hp.smoothing(o, fwhm=np.radians(5.0))
        sl = hp.smoothing(lo, fwhm=np.radians(5.0))
        print(f"{nu:.3f} MHz: ring offset {d[(cen > -30) & (cen < 90)].min():.0f}..{d[(cen > -30) & (cen < 90)].max():.0f} K; "
              f"hybrid = {a:.3f} x LFSM + {b:.0f} K (corr {np.corrcoef(lo[s], o[s])[0, 1]:.3f}); 5-deg smoothed "
              f"hybrid/LFSM median {np.median(sm[s] / sl[s]):.3f}, rms of log ratio {np.std(np.log(sm[s] / sl[s])):.3f}; "
              f"fraction of pixels <= 0: {(o <= 0).mean():.4f}", flush=True)
    np.savez(out, nu_mhz=np.array(NODES), maps_k=np.array(maps), cover=cover.astype(np.float32),
             source="LAMBDA OVRO-LWA (Eastwood et al. 2018), hybrid with LFSM ring means; see prep_ovrolwa.py")
    print("wrote", out)


if __name__ == "__main__":
    mapdir = sys.argv[1]
    main(mapdir, sys.argv[2] if len(sys.argv) > 2 else os.path.join(mapdir, "ovrolwa_hybrid_n256.npz"),
         float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)
