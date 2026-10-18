import numpy as np, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
z = np.load(sys.argv[1], allow_pickle=True); out = sys.argv[2]
t = z['t']; ti = t[::3]
cols = {'none': '#888888', 'tap': '#1f77b4', 'image': '#ff7f0e', 'epical': '#2ca02c'}
lab = {'none': 'cable model only', 'tap': 'voltage tap 2% + StEFCal + Kalman', 'image': 'EPIC images + grid solver + Kalman', 'epical': 'EPICal feedback (Cyg A pixel)'}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for m in ['none', 'tap', 'image', 'epical']:
    ax[0].plot(t, z[f'err_{m}'][:, 1], color=cols[m], label=lab[m])
    ax[1].plot(ti, z[f'img_{m}'], 'o-', ms=3, color=cols[m], label=lab[m])
ax[0].plot(t, z['raw_tap'][:, 1], ':', color=cols['tap'], lw=1, label='tap, per-interval StEFCal (before Kalman)')
for a in ax:
    for x, s in [(300, '4 stands jump 120 deg'), (450, '1 stand dies')]:
        a.axvline(x, color='k', lw=0.6, ls='--'); a.text(x + 5, 0.03, s, fontsize=7, rotation=90, va='bottom', transform=a.get_xaxis_transform())
    a.set_xlabel('time (s)'); a.set_yscale('log'); a.grid(alpha=0.3)
ax[0].set_ylabel('phase error of applied gains, rms (deg)'); ax[1].set_ylabel('image residual |I - I_ideal| / |I_ideal|')
ax[0].legend(fontsize=7, loc='upper right'); ax[0].set_title('Gain error (16 channels, 75.8 MHz band, X pol)', fontsize=9)
ax[1].set_title('EPIC image fidelity vs ideal gains', fontsize=9)
plt.tight_layout(); plt.savefig(out, dpi=110)
for m in ['tap', 'image', 'epical', 'none']:
    e = z[f'err_{m}'][:, 1]
    print(f'{m:7s} phase err median over t>=60s: {np.median(e[t>=60]):.2f} deg; after jump (310-340 s) max {e[(t>=300)&(t<340)].max():.2f}; final img residual {np.median(z[f"img_{m}"][ti>=60]):.3f}; amp+phase {np.median(z[f"err_{m}"][t>=60, 0]):.3f}')
