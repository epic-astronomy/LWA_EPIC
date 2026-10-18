"""How much of a per-antenna direction-dependent correction can an S x S gridding kernel carry?

In EPIC a stand's E-field contribution to image direction l is weighted by
    k_a(l) = sum_cells K_a(cell) exp(-2 pi i delta (cell - pos_a) . l),
the transform of its gridding kernel (K_a: the production 5x5 prolate, w-projected).
A direction-dependent correction c_a(l) needs k'_a(l) = k_a(l) c_a(l). With support S
the kernel response is a 2D trigonometric polynomial of degree (S-1)/2 in delta * l:
over the visible disk (|l| < 1) it can follow only structure on scales of roughly
1 / (delta (S - 1) / 2) in l (35 deg for S = 5 at delta = 0.424).

For random smooth screens c = exp(eps), eps a Gaussian random field over (l, m) with
correlation length theta (deg, in direction-cosine units) and rms 0.1 (rad / log-amp),
this fits the best S x S kernel (same anchor, centred) by least squares over the
visible disk and reports the fraction of the correction captured:
    1 - ||k' - k c|| / ||k - k c||.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402

from streamcal import epicop as E  # noqa: E402

cfg = E.EpicConfig()
rng = np.random.default_rng(1)
n = 96
lg = np.linspace(-1, 1, n)
L, M = np.meshgrid(lg, lg)
vis = L ** 2 + M ** 2 < 0.98
lv, mv = L[vis], M[vis]
nz = np.sqrt(1 - lv ** 2 - mv ** 2)
w = nz  # weight by sin(elevation): the sky near the horizon matters less


def screen(theta_deg, rms=0.1):
    """Complex smooth field: Gaussian random field (FFT-filtered white noise) on the (l, m) grid."""
    s = np.radians(theta_deg)  # correlation length in direction cosine
    k = np.fft.fftfreq(n, d=lg[1] - lg[0])
    KX, KY = np.meshgrid(k, k)
    filt = np.exp(-2 * (np.pi * s) ** 2 * (KX ** 2 + KY ** 2))
    f = [np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(n, n))) * filt)) for _ in range(2)]
    f = [x / x[vis].std() * rms for x in f]
    return np.exp(f[0][vis] + 1j * f[1][vis])


def basis(S, off):
    """Image response of each cell of an S x S kernel centred on the stand (offset off in cells)."""
    h = S // 2
    cells = [(dx, dy) for dy in range(-h, h + 1) for dx in range(-h, h + 1)]
    B = np.stack([np.exp(-2j * np.pi * cfg.delta * ((dx - off[0]) * lv + (dy - off[1]) * mv)) for dx, dy in cells], 1)
    return B, cells


def main():
    out = []
    for chan in (1440, 3168):
        kc, xp, yp, pos, used = E.kernels(cfg, chan)
        stands = np.flatnonzero(used)[::23]
        for theta in (5, 10, 20, 40, 80):
            caps = {S: [] for S in (5, 7, 9, 11)}
            for a in stands:
                # production kernel response relative to the stand position
                ox, oy = xp[a] - pos[a, 0], yp[a] - pos[a, 1]
                ka = (np.exp(-2j * np.pi * cfg.delta * (np.outer(lv, ox) + np.outer(mv, oy))) @ kc[a])
                c = screen(theta)
                target = ka * c
                for S in caps:
                    # anchor: the production kernel's centre cell; fractional offset of the stand from it
                    cx, cy = xp[a][len(xp[a]) // 2], yp[a][len(yp[a]) // 2]
                    B, _ = basis(S, (pos[a, 0] - cx, pos[a, 1] - cy))
                    coef = np.linalg.lstsq(B * np.sqrt(w)[:, None], target * np.sqrt(w), rcond=None)[0]
                    fit = B @ coef
                    caps[S].append(1 - np.linalg.norm((fit - target) * np.sqrt(w)) / np.linalg.norm((ka - target) * np.sqrt(w)))
            line = "  ".join(f"S={S}: {np.median(v):5.2f}" for S, v in caps.items())
            print(f"{chan * E.CHAN_BW / 1e6:5.1f} MHz  screen scale {theta:3d} deg  captured fraction  {line}", flush=True)
            out.append((chan, theta, {S: float(np.median(v)) for S, v in caps.items()}))
    np.save("dd_kernel_fit.npy", out, allow_pickle=True)


if __name__ == "__main__":
    main()
