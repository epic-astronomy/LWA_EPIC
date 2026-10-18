import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streamcal import sky as skymodel, epicop, forward, solvers as calib  # noqa: E402,F401
sys.modules.update({"skymodel": skymodel, "epicop": epicop, "forward": forward, "calib": calib})
import numpy as np, sys, glob
from astropy.time import Time
import forward as F, calib as C, epicop as E, tbf_cal as TC
files = sorted(glob.glob('../tbfR_[0-9].npz'))
ant = np.flatnonzero(E.eg.used_stands())
xyz = E.eg.stand_xyz()[ant]
bl = np.linalg.norm(xyz[:, None, :2] - xyz[None, :, :2], axis=-1)
allsol, allchi, times = [], [], []
for fn in files:
    z = np.load(fn); R = z['R']; chans = z['chans']; T = int(z['nsamp']); t = Time(str(z['t']))
    freq = chans * 25e3
    sol = np.full((2, len(freq), len(ant)), np.nan, complex); chi = np.full((2, len(freq)), np.nan); rho = np.zeros((2, len(freq)))
    for p, pol in enumerate(('XX', 'YY')):
        snap = F.build_snapshot(t, freq / E.CHAN_BW, pol=pol, nside=64)
        for ci, f in enumerate(freq):
            Rc = R[ci, p][np.ix_(ant, ant)]
            ac = np.real(np.diag(Rc)); okc = (ac > 0.3*np.median(ac)) & (ac < 3*np.median(ac))
            cc = TC.cable_correction(f, p)[ant]
            Rc = cc[:, None] * Rc * np.conj(cc)[None, :]
            M = snap.matrix(ant, ci)
            W = ((bl / (299792458.0 / f)) >= 1.0).astype(float); np.fill_diagonal(W, 0); W *= np.outer(okc, okc)
            g, _, _ = C.stefcal(Rc, M, W, niter=300)
            d = np.maximum(np.real(np.diag(Rc)), 1e-12)
            r2 = np.abs(Rc - np.outer(g, g.conj()) * M) ** 2 / (np.outer(d, d) / T) * W
            chi[p, ci] = np.median((r2.sum(1) / np.maximum(W.sum(1), 1))[okc])
            rho[p, ci] = np.abs(np.vdot(M*W, Rc*W))/np.linalg.norm(M*W)/np.linalg.norm(Rc*W)
            sol[p, ci] = np.where(okc, g, np.nan)
    allsol.append(sol); allchi.append(chi); times.append(t)
    lst = t.sidereal_time('apparent', longitude=-106.886).hour
    print(f'{t.isot} LST {lst:5.2f}h  chi2/dof per chan (X): ' + ' '.join(f'{c:.1f}' if c < 100 else '>99' for c in chi[0]))
    print(f'{"":30s} corr (X):        ' + ' '.join(f'{c:.2f}' for c in rho[0]))
allsol = np.array(allsol); allchi = np.array(allchi)
np.savez('../tbf_stability.npz', sol=allsol, chi=allchi, t=[t.isot for t in times], chans=chans, ant=ant)
good_ch = np.all(allchi[:, 0] < 3, axis=0) & np.all(allchi[:, 1] < 3, axis=0)
print('channels good in all files:', chans[good_ch])
# stability: phase/amp change relative to file 0, common phase removed per channel
for k in range(1, len(files)):
    for p in range(2):
        r = allsol[k, p][good_ch] / allsol[0, p][good_ch]
        r = r * np.conj(np.nansum(r, 1, keepdims=True)) / np.abs(np.nansum(r, 1, keepdims=True))
        # remove a per-antenna delay (cable temperature) and look at what is left
        dph = np.degrees(np.nanstd(np.angle(r)))
        damp = np.nanstd(np.log(np.abs(r)))
        print(f'  {times[k].isot} vs {times[0].isot[11:16]} pol {"XY"[p]}: phase change rms {dph:5.2f} deg, amp change rms {100*damp:4.1f}%')
