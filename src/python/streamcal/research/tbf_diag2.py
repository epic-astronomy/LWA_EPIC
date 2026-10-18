import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streamcal import sky as skymodel, epicop, forward, solvers as calib  # noqa: E402,F401
sys.modules.update({"skymodel": skymodel, "epicop": epicop, "forward": forward, "calib": calib})
import numpy as np, sys
from astropy.time import Time
import forward as F, calib as C, epicop as E, tbf_cal as TC
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
z = np.load(sys.argv[1]); R = z['R']; chans = z['chans']; T = int(z['nsamp']); t = Time(str(z['t']))
freq = chans * 25e3
ant = np.flatnonzero(E.eg.used_stands())
xyz = E.eg.stand_xyz()[ant]
bl = np.linalg.norm(xyz[:, None, :2] - xyz[None, :, :2], axis=-1)
p = 0
snap = F.build_snapshot(t, freq / E.CHAN_BW, pol='XX', nside=64)
ac = np.real(np.einsum('cii->ci', R[:, p]))[:, ant]
okc = ac > 0.3 * np.median(ac, axis=1, keepdims=True)
okc &= ac < 3 * np.median(ac, axis=1, keepdims=True)
sols = np.zeros((len(freq), len(ant)), complex); chi = np.zeros((len(freq), len(ant)))
for ci, f in enumerate(freq):
    Rc = R[ci, p][np.ix_(ant, ant)]
    cc = TC.cable_correction(f, p)[ant]
    Rc = cc[:, None] * Rc * np.conj(cc)[None, :]
    M = snap.matrix(ant, ci)
    W = ((bl / (299792458.0 / f)) >= 1.0).astype(float); np.fill_diagonal(W, 0)
    W *= np.outer(okc[ci], okc[ci])
    g, _, _ = C.stefcal(Rc, M, W, niter=300)
    d = np.maximum(np.real(np.diag(Rc)), 1e-12)
    r2 = np.abs(Rc - np.outer(g, g.conj()) * M) ** 2 / (np.outer(d, d) / T) * W
    chi[ci] = r2.sum(1) / np.maximum(W.sum(1), 1)
    sols[ci] = np.where(okc[ci], g, np.nan)
    # data-model correlation per channel
    rho = np.abs(np.vdot(M*W, Rc*W))/np.linalg.norm(M*W)/np.linalg.norm(Rc*W)
    if ci % 5 == 0: print(f'chan {chans[ci]} {f/1e6:.2f} MHz: chi2/dof median {np.median(chi[ci][okc[ci]]):.3f}; data-model corr (cable only) {rho:.3f}; ok ants {okc[ci].sum()}')
ref = np.nanargmax(np.nansum(np.abs(sols), 0))
ph = np.angle(sols * np.conj(sols[:, [ref]]))
fig, ax = plt.subplots(2, 3, figsize=(12, 6))
for k, a in enumerate(rng := np.random.default_rng(0).choice(len(ant), 6, replace=False)):
    ax.flat[k].plot(freq/1e6, np.degrees(ph[:, a]), 'o-'); ax.flat[k].set_title(f'stand idx {ant[a]}'); ax.flat[k].set_ylim(-190, 190)
plt.tight_layout(); plt.savefig('../tbf_phase.png', dpi=70)
amp = np.abs(sols)
print('amp spread per channel (median over chans):', np.nanmedian(np.nanstd(amp, 1)/np.nanmean(amp, 1)))
