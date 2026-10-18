import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streamcal import sky as skymodel, epicop, forward, solvers as calib  # noqa: E402,F401
sys.modules.update({"skymodel": skymodel, "epicop": epicop, "forward": forward, "calib": calib})
import numpy as np
z = np.load('../tbf_stability.npz', allow_pickle=True); sol, chi, t, chans = z['sol'], z['chi'], z['t'], z['chans']
freq = chans * 25e3
for k in [1, 2, 3]:
    for p in range(2):
        good = chi[k, p] < 3
        e = sol[k, p][good]
        ok = np.all(np.isfinite(e), 0)
        e = e[:, ok]
        e = e * np.conj(np.sum(e, 1, keepdims=True)) / np.abs(np.sum(e, 1, keepdims=True))
        ph = np.unwrap(np.angle(e), axis=0)
        A = np.c_[np.ones(good.sum()), 2*np.pi*(freq[good]-freq[good].mean())]
        coef = np.linalg.lstsq(A, ph, rcond=None)[0]
        amp = np.abs(e) / np.median(np.abs(e), axis=1, keepdims=True)
        # coherence of an uncalibrated (cable-model only) beam/image: |sum eps|^2 / (N sum |eps|^2), per channel
        eta = np.abs(e.sum(1))**2 / (e.shape[1] * (np.abs(e)**2).sum(1))
        print(f'{t[k][11:16]} pol {"XY"[p]}: residual (cable model -> solved) phase rms (MAD) {np.degrees(1.4826*np.median(np.abs(np.angle(e)))):5.1f} deg, '
              f'delay rms (MAD) {1.4826*np.median(np.abs(coef[1]-np.median(coef[1])))*1e9:4.1f} ns, amp rms (MAD) {100*1.4826*np.median(np.abs(np.log(amp)-np.median(np.log(amp)))):4.1f}%, inputs <0.3 median {np.mean(np.median(amp,0)<0.3)*100:.1f}%; '
              f'point-source coherence of cable-only weights {np.median(eta):.3f}')
