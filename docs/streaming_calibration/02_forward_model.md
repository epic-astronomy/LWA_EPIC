# 2. Forward model of the sky

The calibration solves for gains against a predicted covariance $M(f,t)$. Its accuracy sets the bias floor of the solution (section 3.2), so this section records the components and the choices made.

## 2.1 Components

| Component | Model | Notes |
|---|---|---|
| Cyg A | Baars et al. 1977: $\log S = 4.695 + 0.085\log\nu - 0.178\log^2\nu$ (ν in MHz) | 24.3 kJy at 38 MHz, 17.1 kJy at 74 MHz |
| Cas A | Baars et al. 1977 (epoch 1965) with secular decrease $0.97 - 0.30\log(\nu/\mathrm{GHz})$ % per year, to epoch 2026.8 | 15.4 kJy at 38 MHz, 10.1 kJy at 74 MHz. The Baars rate overestimates the low-frequency fading; de Gasperin et al. 2020 should replace it |
| Tau A | 1.6 kJy at 74 MHz, $\nu^{-0.25}$ | Flattening below 100 MHz |
| Vir A | 2.1 kJy at 74 MHz, $\nu^{-0.8}$ | VLSS scale |
| Sun | Quiet Sun, 1.5×10⁴ Jy at 74 MHz, $\propto\nu$ | **Flux solved** in every interval; the daytime simulation with the Sun at 3× the model recovered 2.97–3.04 |
| Diffuse sky | LFSM (Dowell et al. 2017) through pygdsm, Galactic HEALPix, evaluated at three frequency nodes per band and interpolated log-log | nside 64 (0.92°) is the minimum for the 128-grid baselines (section 3.1) |
| Beam | `lsl.sim.beam.beam_response('empirical', 'XX' or 'YY', az, alt, f)` | Evaluated once per frequency for all directions |

Point sources below 5° elevation are dropped. Above the horizon at nside 64 there are about 24,500 diffuse pixels.

## 2.2 Evaluation

$$M = \sum_{\rm comp} A_{\rm comp}\,\mathrm{diag}(B\odot S)_{\rm comp}\,A_{\rm comp}^H,\qquad A_{as} = e^{2\pi i f\,\mathbf r_a\cdot\hat s/c}$$

Implementation choices:

- **Geometry:** full 3D stand positions (east, north, up); no w approximation enters the model.
- **Arithmetic:** geometric delays are cached per component, and the phases are evaluated as float32 sin/cos (relative error of $M$ 2.5×10⁻⁷). Each channel needs one complex64 GEMM.
- **Cost:** 32 solved channels × 2 polarisations take 12 s on a Xeon Silver 4210 with 8 threads, refreshed every 60 s. In 60 s the sky moves 0.25°, an eighth of the finest fringe at 75 MHz.

## 2.3 What the model does not contain

- **Unresolved and small-scale diffuse structure** below the LFSM resolution (2–5°). It is simulated as 10% multiplicative structure and costs 0.4° in the covariance solution (2.2° in the image solution).
- **Faint compact sources** (3C and AARTFAAC-catalogue sources of 100–800 Jy). These cost about 0.2°.
- **Ionospheric refraction of the A-team** (about 0.1°). It is absorbed into the direction-independent gains. The remedy is per-source position nuisance parameters (section 4.7).
- **Polarisation.** The sky is treated as unpolarised, with per-dipole power patterns.
- **RFI.** It is handled by the χ² gate rather than modelled.
- **Mutual coupling and embedded element patterns.** All stands share one beam, a second-order effect at about 5 m spacing (Jones & Wayth 2021).

On LWA-SV data the residual χ²/dof of 1.1–1.9 in clean channels (section 3.6) bounds the combined effect of these omissions at roughly the level of the thermal noise of a 0.4 s, 25 kHz snapshot.
