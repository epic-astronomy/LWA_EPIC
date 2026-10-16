import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streamcal import sky as skymodel, epicop, forward, solvers as calib  # noqa: E402,F401
sys.modules.update({"skymodel": skymodel, "epicop": epicop, "forward": forward, "calib": calib})
import numpy as np
z = np.load('../tbf_stability.npz', allow_pickle=True); sol, chi, t, chans = z['sol'], z['chi'], z['t'], z['chans']
freq = chans * 25e3
use = [1, 2, 3]
good = np.all(chi[use] < 3, axis=(0, 1))
print('clean channels (MHz):', np.round(freq[good] / 1e6, 1))
for k in [2, 3]:
    for p in range(2):
        a, b = sol[1, p][good], sol[k, p][good]
        r = b / a
        ok = np.all(np.isfinite(r), 0)
        r = r[:, ok]
        r = r * np.conj(np.sum(r, 1, keepdims=True)) / np.abs(np.sum(r, 1, keepdims=True))
        ph = np.angle(r)
        # fit per-antenna delay change + phase offset change
        A = np.c_[np.ones(good.sum()), 2*np.pi*(freq[good]-freq[good].mean())]
        coef = np.linalg.lstsq(A, ph, rcond=None)[0]
        resid = ph - A @ coef
        print(f'{t[k][11:16]} vs {t[1][11:16]} pol {"XY"[p]}: phase change rms {np.degrees(ph.std()):5.2f} deg '
              f'(offset part {np.degrees(coef[0].std()):.2f} deg, delay part {coef[1].std()*1e9:.2f} ns, residual {np.degrees(resid.std()):.2f} deg); '
              f'amp change rms {100*np.std(np.log(np.abs(r))):.1f}%  [{ok.sum()} ants]')
# single-file noise floor: phase scatter between adjacent channels
for k in use:
    for p in range(1):
        s = sol[k, p][good]
        d = np.angle(s[1:] * np.conj(s[:-1]))
        d = d - np.angle(np.nansum(np.exp(1j*d), 1, keepdims=True))
        print(f'{t[k][11:16]} pol X: adjacent-channel phase difference rms {np.degrees(np.nanstd(d))/np.sqrt(2):.2f} deg per channel (noise + 0.5 MHz structure)')
