"""Build the compact-source tables of the calibration sky model (streamcal/data/).

    python build_sky_catalogs.py DEGASPERIN_DIR VLSSR_TSV OUTDIR

1. A-team spectra from de Gasperin et al. 2020 (A&A 635, A150; CDS J/A+A/635/A150).
   The low-resolution clean-component lists (xxxLow-sources.txt, WSClean format,
   ordinary polynomial spectra S = I + sum_k c_k (nu/nu0 - 1)^(k+1)) are summed into
   integrated flux densities, tabulated at 30-77 MHz (the range of the LOFAR LBA data;
   outside it the model is extrapolated by sky.py with the end slope). Epoch 2015-08-26.
   Fetch: https://cdsarc.cds.unistra.fr/ftp/J/A+A/635/A150/{Cas,Cyg,Tau}Low-sources.txt and
   VirLow-sources.txt.gz.
2. Bright compact sources from VLSSr (Lane et al. 2014, 74 MHz, VizieR VIII/97):
   components above 15 Jy, merged friends-of-friends within 0.5 deg (LWA-SV resolves
   2-4 deg), flux-weighted positions, sources within 3 deg of an A-team source dropped
   (they are A-team components or artefacts). Fetch:
   https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=VIII/97/catalog&-out=RAJ2000,DEJ2000,Sp,MajAx,MinAx&Sp=%3E15&-out.max=unlimited
"""
import os
import re
import sys

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

dg_dir, vlssr_tsv, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)

# 1. de Gasperin 2020 integrated spectra
nus = np.arange(30.0, 77.01, 1.0)
rows = []
for name, fn in (("CasA", "CasLow-sources.txt"), ("CygA", "CygLow-sources.txt"), ("TauA", "TauLow-sources.txt"),
                 ("VirA", "VirLow-sources.txt")):
    nu0, comps = None, []
    for line in open(os.path.join(dg_dir, fn)):
        if line.startswith("Format"):
            nu0 = float(re.search(r"ReferenceFrequency=.([0-9.]+)", line).group(1))
            continue
        if not line.strip():
            continue
        f = line.split(",")
        si = [float(x) for x in re.search(r"\[(.*?)\]", line).group(1).split(",")]
        assert line.split("]")[1].split(",")[1].strip() == "false", "logarithmic spectra not handled"
        comps.append((float(f[4]), si))
    x = nus * 1e6 / nu0 - 1
    S = np.array([sum(I + sum(c * xx ** (k + 1) for k, c in enumerate(si)) for I, si in comps) for xx in x])
    rows.append((name, S))
    print(f"{name}: {len(comps)} components, S(38) = {np.interp(38, nus, S):.0f} Jy, S(74) = {np.interp(74, nus, S):.0f} Jy")
with open(os.path.join(out, "ateam_degasperin2020.csv"), "w") as fh:
    fh.write("# Integrated flux density (Jy) of the A-team from the de Gasperin et al. 2020 (A&A 635, A150)\n"
             "# low-resolution clean components (CDS J/A+A/635/A150), epoch 2015-08-26. Built by\n"
             "# research/build_sky_catalogs.py.\n")
    fh.write("nu_mhz," + ",".join(n for n, _ in rows) + "\n")
    for i, nu in enumerate(nus):
        fh.write(f"{nu:.1f}," + ",".join(f"{S[i]:.1f}" for _, S in rows) + "\n")

# 2. VLSSr bright sources, merged
ra, dec, sp = [], [], []
for line in open(vlssr_tsv):
    p = line.rstrip("\n").split("\t")
    if line.startswith("#") or len(p) < 3 or not re.match(r"^\d", p[0].strip()):
        continue
    c = SkyCoord(p[0].strip(), p[1].strip(), unit=(u.hourangle, u.deg))
    ra.append(c.ra.deg)
    dec.append(c.dec.deg)
    sp.append(float(p[2]))
c = SkyCoord(np.array(ra) * u.deg, np.array(dec) * u.deg)
sp = np.array(sp)
ateam = SkyCoord(["19h59m28.36s +40d44m02.1s", "23h23m24.0s +58d48m54s", "05h34m31.94s +22d00m52.2s",
                  "12h30m49.42s +12d23m28.0s"], unit=(u.hourangle, u.deg))
near = np.zeros(len(c), bool)
for a in ateam:
    near |= c.separation(a).deg < 3.0
c, sp = c[~near], sp[~near]
group = -np.ones(len(c), int)
g = 0
for i in np.argsort(-sp):
    if group[i] >= 0:
        continue
    stack = [i]
    group[i] = g
    while stack:
        j = stack.pop()
        for k in np.flatnonzero((group < 0) & (c.separation(c[j]).deg < 0.5)):
            group[k] = g
            stack.append(k)
    g += 1
merged = []
for k in range(g):
    m = group == k
    w = sp[m]
    v = np.sum(c[m].cartesian.xyz.value * w, axis=1)
    cc = SkyCoord(x=v[0], y=v[1], z=v[2], representation_type="cartesian").spherical
    merged.append((cc.lon.deg, cc.lat.deg, w.sum(), int(m.sum())))
merged.sort(key=lambda r: -r[2])
with open(os.path.join(out, "vlssr_bright.csv"), "w") as fh:
    fh.write("# VLSSr (Lane et al. 2014, VizieR VIII/97) components above 15 Jy at 74 MHz, merged within\n"
             "# 0.5 deg, A-team regions (3 deg) excluded. Built by research/build_sky_catalogs.py.\n")
    fh.write("ra_deg,dec_deg,s74_jy,ncomp\n")
    for r in merged:
        fh.write(f"{r[0]:.4f},{r[1]:.4f},{r[2]:.2f},{r[3]}\n")
print(f"VLSSr: {len(sp)} components outside the A-team regions -> {len(merged)} sources; "
      f"brightest: " + ", ".join(f"{r[2]:.0f} Jy ({r[0]:.1f},{r[1]:+.1f})" for r in merged[:8]))
print(f"  above 30 Jy: {sum(r[2] > 30 for r in merged)}, above 100 Jy: {sum(r[2] > 100 for r in merged)}")
