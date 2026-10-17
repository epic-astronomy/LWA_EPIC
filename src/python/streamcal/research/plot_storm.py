"""Plot sim_storm.py output: applied-gain phase error (tip/tilt removed) and residual image shift."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

z = np.load(sys.argv[1], allow_pickle=True)
t = z["t"]
args = eval(str(z["args"]))
lab = {"A": "A: Sun flux only", "B": "B: all fluxes", "C": "C: fluxes + positions", "D": "D: C + 1/f^2 band fit"}
col = {"A": "#d62728", "B": "#ff7f0e", "C": "#1f77b4", "D": "#2ca02c"}
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for v in "ABCD":
    e = z[f"err_{v}"]
    ax[0].plot(t, e[:, 1], color=col[v], label=lab[v], ls="--" if v == "D" else "-")
    ax[1].plot(t, e[:, 2], color=col[v], label=lab[v], ls="--" if v == "D" else "-")
for a in ax:
    b0, b1, _ = (float(x) for x in args["burst"].split(":"))
    s0, s1, _ = (float(x) for x in args["absorb"].split(":"))
    if b1 > b0:
        a.axvspan(b0, b1, color="orange", alpha=0.15, label="solar burst")
    a.axvspan(s0, s1, color="grey", alpha=0.15, label="absorption")
    a.set_xlabel("time (s)")
    a.grid(alpha=0.3)
f = z["freq"].mean() / 1e6
ax[0].set_ylabel("phase error of applied gains, tip/tilt removed (deg)")
ax[1].set_ylabel("residual image shift (deg)")
ax[0].set_title(f"Ionospheric storm, {f:.1f} MHz: refraction {args['refr_deg']} deg x (74/f)^2, scintillation {args['scint']}", fontsize=9)
ax[1].set_title("Position error left in the images", fontsize=9)
ax[0].legend(fontsize=7)
plt.tight_layout()
plt.savefig(sys.argv[2], dpi=110)
for v in "ABCD":
    e = z[f"err_{v}"]
    q = t >= 60
    print(f"{v}: phase median {np.median(e[q, 1]):.2f} max {e[q, 1].max():.2f} deg; shift median {np.median(e[q, 2]):.3f} max {e[q, 2].max():.3f} deg")
