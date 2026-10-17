"""Inspect an OVRO-LWA map against LFSM: units, coverage, large scales, A-team residuals."""
import sys
import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
from pygdsm import LowFrequencySkyModel

fn, nu = sys.argv[1], float(sys.argv[2])
m = hp.read_map(fn, dtype=np.float64)
print("nside", hp.npix2nside(m.size), "finite", np.isfinite(m).mean(), "unseen", (m == hp.UNSEEN).mean(),
      "zero", (m == 0).mean())
bad = ~np.isfinite(m) | (m == hp.UNSEEN)
print("percentiles", np.percentile(m[~bad], [0.1, 1, 10, 50, 90, 99, 99.9]))
NS = 128
mo = hp.ud_grade(np.where(bad, hp.UNSEEN, m), NS)
lf = hp.ud_grade(LowFrequencySkyModel(freq_unit="MHz").generate(nu), NS)
th, ph = hp.pix2ang(NS, np.arange(hp.nside2npix(NS)))
g = SkyCoord(l=ph * u.rad, b=(np.pi / 2 - th) * u.rad, frame="galactic").icrs
dec = g.dec.deg
ok = mo != hp.UNSEEN
print("coverage nside128", ok.mean(), "min dec covered", dec[ok].min())
for lo in range(-30, 90, 15):
    s = ok & (dec >= lo) & (dec < lo + 15)
    print(f" dec {lo:+3d}..{lo + 15:+3d}: ovro median {np.median(mo[s]):8.0f}  lfsm median {np.median(lf[s]):8.0f}  "
          f"ratio of medians {np.median(mo[s]) / np.median(lf[s]):.3f}  corr {np.corrcoef(mo[s], lf[s])[0, 1]:.3f}")
# smoothed comparison at 5 deg
both = ok & (dec > -25)
sm_o = hp.smoothing(np.where(ok, mo, 0), fwhm=np.radians(5))
sm_w = hp.smoothing(ok.astype(float), fwhm=np.radians(5))
sm_l = hp.smoothing(lf, fwhm=np.radians(5))
r = sm_o / np.maximum(sm_w, 1e-3) / sm_l
s = both & (sm_w > 0.9)
print("5-deg smoothed ovro/lfsm: median %.3f  16-84%% %.3f %.3f" % (np.median(r[s]), *np.percentile(r[s], [16, 84])))
lin = np.polyfit(sm_l[s], (sm_o / np.maximum(sm_w, 1e-3))[s], 1)
print("linear fit ovro = a*lfsm + b:", lin)
# A-team residuals at native res
NSn = hp.npix2nside(m.size)
for name, c in [("CygA", SkyCoord("19h59m28.36s", "+40d44m02.1s")), ("CasA", SkyCoord("23h23m24.0s", "+58d48m54s")),
                ("TauA", SkyCoord("05h34m31.94s", "+22d00m52.2s")), ("VirA", SkyCoord("12h30m49.42s", "+12d23m28.0s")),
                ("HerA", SkyCoord("16h51m08.1s", "+04d59m33s")), ("3C123", SkyCoord("04h37m04.4s", "+29d40m14s")),
                ("HyaA", SkyCoord("09h18m05.7s", "-12d05m44s")), ("3C353", SkyCoord("17h20m28.2s", "-00d58m47s"))]:
    v = hp.ang2vec(c.galactic.l.deg, c.galactic.b.deg, lonlat=True)
    out = []
    for r1 in (0.5, 1, 2, 4):
        d = hp.query_disc(NSn, v, np.radians(r1))
        dd = d[~bad[d]]
        out.append(f"r<{r1}: med {np.median(m[dd]):7.0f} max {m[dd].max():7.0f} min {m[dd].min():7.0f}")
    print(name, " | ".join(out))
# power spectra ratio
cl_o = hp.anafast(np.where(ok, mo - mo[ok].mean(), 0), lmax=200)
cl_l = hp.anafast(np.where(ok, lf - lf[ok].mean(), 0), lmax=200)
for l in (1, 2, 4, 8, 16, 32, 64, 100, 150, 200):
    print(f" l={l:3d} C_ovro/C_lfsm {cl_o[l] / cl_l[l]:.3f}")
