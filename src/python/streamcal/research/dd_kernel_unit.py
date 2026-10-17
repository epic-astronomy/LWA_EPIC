"""Unit checks of direction-dependent corrections through the gridding kernels (section 8.7).

DI gains applied through kernels (exact), a linear phase screen corrected through 5x5
kernels (point-source peak, whole-sky error, fit residual by radius).
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from astropy.time import Time
from streamcal import epicop as E, forward as F, solvers as C
cfg = E.EpicConfig(); chan = 1440; f = chan*E.CHAN_BW
op0 = E.build_gridop(cfg, chan); ant = op0.ant; n = len(ant)
r = F.stand_enu()[ant]; phi = F.geometric_phase(chan, ant)
snap = F.build_snapshot(Time('2021-04-04T11:35:00'), [chan], nside=32)
enu = np.concatenate([c.enu for c in snap.comps]); S = np.concatenate([c.flux[:, 0] for c in snap.comps])
kc, xp, yp, pos, used = E.kernels(cfg, chan)
rng = np.random.default_rng(1)
def vis(J):
    A = np.exp(2j*np.pi*f/299792458.0*(r@enu.T)) * J
    return (A*S) @ A.conj().T
def img(op, R, w): return E.image_from_vis(op, w[:,None]*R*np.conj(w)[None,:], with_autos=False)
ideal = img(op0, vis(1.0), phi)
def err(x): sc = np.sum(x*ideal)/np.sum(x*x); return np.linalg.norm(sc*x-ideal)/np.linalg.norm(ideal)
g = np.exp(rng.normal(0,0.15,n) + 1j*rng.uniform(-np.pi,np.pi,n))
Rg = vis(g[:,None])
kc1 = kc.copy(); kc1[ant] = kc[ant] / g[:,None]
print('DI gains, no correction', err(img(op0, Rg, phi)))
print('DI gains, corrected by weights', err(img(op0, Rg, phi/g)))
print('DI gains, corrected by kernels kc/g', err(img(E.build_gridop(cfg, chan, kc_override=kc1), Rg, phi)))
# small linear-phase screen per stand: J_a(s) = exp(i (alpha_a l + beta_a m)) with 5 deg rms at |l|=0.5
al = rng.normal(0, np.radians(10), (n, 2))
J = np.exp(1j*(al[:,0:1]*enu[None,:,0] + al[:,1:2]*enu[None,:,1]))
RJ = vis(J)
print('screen, no correction', err(img(op0, RJ, phi)))
for sgn in (+1, -1):
    kc2 = kc.copy()
    L, M = np.meshgrid(np.linspace(-1,1,96), np.linspace(-1,1,96)); v = L**2+M**2<0.98; lv, mv = L[v], M[v]
    for i, a in enumerate(ant):
        B = np.exp(-2j*np.pi*cfg.delta*(np.outer(lv, xp[a]-pos[a,0]) + np.outer(mv, yp[a]-pos[a,1])))
        corr = np.exp(-1j*sgn*(al[i,0]*lv + al[i,1]*mv))
        kc2[a] = np.linalg.lstsq(B, (B@kc[a])*corr, rcond=None)[0]
    print(f'screen, kernels with correction sign {sgn:+d}', err(img(E.build_gridop(cfg, chan, kc_override=kc2), RJ, phi)))
# --- diagnose: implemented correction at one direction, through the actual operator
p, Fr, _ = E.eg.array_frame()
s = np.array([0.3, 0.1, 0]); s[2] = np.sqrt(1 - s[0]**2 - s[1]**2)
sg = Fr @ s
kc2 = kc.copy()
L, M = np.meshgrid(np.linspace(-1,1,96), np.linspace(-1,1,96)); v = L**2+M**2<0.98; lv, mv = L[v], M[v]
for i, a in enumerate(ant):
    B = np.exp(-2j*np.pi*cfg.delta*(np.outer(lv, xp[a]-pos[a,0]) + np.outer(mv, yp[a]-pos[a,1])))
    kc2[a] = np.linalg.lstsq(B, (B@kc[a])*np.exp(-1j*(al[i,0]*lv + al[i,1]*mv)), rcond=None)[0]
def resp(k, a, d):
    return np.exp(-2j*np.pi*cfg.delta*((xp[a]-pos[a,0])*d[0] + (yp[a]-pos[a,1])*d[1])) @ k
for lab, d in (('ENU coords', s), ('grid-frame coords', sg)):
    imp = np.array([resp(kc2[a], a, d)/resp(kc[a], a, d) for a in ant])
    want = np.exp(-1j*(al[:,0]*s[0] + al[:,1]*s[1]))
    q = imp/want
    print(f'{lab}: implemented/intended correction at s: phase rms {np.degrees(np.angle(q/np.mean(q))).std():.2f} deg, |.| spread {np.abs(q).std():.3f}')
# single point source test through the operator
R1 = C.point_matrix(r, f, s, 1.0)
J1 = np.exp(1j*(al[:,0]*s[0] + al[:,1]*s[1]))
R1J = J1[:,None]*R1*np.conj(J1)[None,:]
i0 = img(op0, R1, phi); i1 = img(op0, R1J, phi); i2 = img(E.build_gridop(cfg, chan, kc_override=kc2), R1J, phi)
print('point source: peak ideal', i0.max(), ' screened', i1.max(), ' kernel-corrected', i2.max())
# --- error over the visible sky only, and the fit residual vs radius
kk = np.fft.fftfreq(op0.N)*op0.N; lp = kk/(op0.N*cfg.delta); LL, MM = np.meshgrid(lp, lp, indexing='ij')
rr = np.hypot(LL, MM)
def err_vis(x, rmax):
    m = rr < rmax; sc = np.sum(x[m]*ideal[m])/np.sum(x[m]**2); return np.linalg.norm(sc*x[m]-ideal[m])/np.linalg.norm(ideal[m])
icor = img(E.build_gridop(cfg, chan, kc_override=kc2), RJ, phi); iunc = img(op0, RJ, phi)
for rmax in (0.5, 0.8, 0.95, 1.0, 2.0):
    print(f'r < {rmax}: error uncorrected {err_vis(iunc, rmax):.4f}, kernel-corrected {err_vis(icor, rmax):.4f}')
a = ant[0]
B = np.exp(-2j*np.pi*cfg.delta*(np.outer(lv, xp[a]-pos[a,0]) + np.outer(mv, yp[a]-pos[a,1])))
tgt = (B@kc[a])*np.exp(-1j*(al[0,0]*lv + al[0,1]*mv)); res = np.abs(B@kc2[a]-tgt)/np.abs(B@kc[a]).max()
rv = np.hypot(lv, mv)
for lo, hi in ((0,0.5),(0.5,0.8),(0.8,0.99)):
    m = (rv>=lo)&(rv<hi); print(f'fit residual / peak response, {lo}-{hi}: {res[m].mean():.4f}; |k| there {np.abs(B@kc[a])[m].mean()/np.abs(B@kc[a]).max():.3f}')
