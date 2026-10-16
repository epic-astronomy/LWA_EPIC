import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
z = np.load('../tbf_stability.npz', allow_pickle=True); sol, chi, t, chans = z['sol'], z['chi'], z['t'], z['chans']
f = chans * 25e3 / 1e6
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for k in range(len(t)):
    ax[0].semilogy(f, chi[k, 0], 'o-', ms=3, label=f'{t[k][11:16]} UTC')
ax[0].axhline(1, color='k', lw=0.6); ax[0].axhline(3, color='r', lw=0.6, ls='--')
ax[0].set_xlabel('frequency (MHz)'); ax[0].set_ylabel('median chi^2/dof (X pol)'); ax[0].legend(fontsize=7)
ax[0].set_title('Fit of forward-modelled sky to LWA-SV TBF covariance\n(1 = thermal noise; dashed = rejection threshold)', fontsize=8)
k = 2
good = chi[k, 0] < 3
e = sol[k, 0]
e = e * np.conj(np.nansum(e, 1, keepdims=True)) / np.abs(np.nansum(e, 1, keepdims=True))
rng = np.random.default_rng(1)
for a in rng.choice(np.flatnonzero(np.all(np.isfinite(e[good]), 0)), 8, replace=False):
    ax[1].plot(f[good], np.degrees(np.angle(e[good, a])), 'o-', ms=3)
ax[1].set_xlabel('frequency (MHz)'); ax[1].set_ylabel('solved phase relative to cable model (deg)')
ax[1].set_title(f'StEFCal solutions, 8 random stands, {t[k][11:16]} UTC, 0.4 s', fontsize=8)
# stability 02 -> 06 UTC
use = np.all(chi[[1, 2, 3], 0] < 3, axis=0)
r = sol[3, 0][use] / sol[1, 0][use]
ok = np.all(np.isfinite(r), 0); r = r[:, ok]
r = r * np.conj(np.sum(r, 1, keepdims=True)) / np.abs(np.sum(r, 1, keepdims=True))
ax[2].hist(np.degrees(np.angle(r)).ravel(), bins=60, range=(-60, 60), alpha=0.7, label='phase change 02->06 UTC')
d = sol[1, 0][use][1:] * np.conj(sol[1, 0][use][:-1]); d = d[:, ok]
d = d * np.conj(np.sum(d, 1, keepdims=True)) / np.abs(np.sum(d, 1, keepdims=True))
ax[2].hist(np.degrees(np.angle(d)).ravel() / np.sqrt(2) * np.sqrt(2), bins=60, range=(-60, 60), alpha=0.5, label='adjacent-channel difference (noise ref.)')
ax[2].set_xlabel('deg'); ax[2].legend(fontsize=7); ax[2].set_title('Gain stability over 4 h (X pol, all clean channels)', fontsize=8)
plt.tight_layout(); plt.savefig('../tbf_realdata.png', dpi=110)
