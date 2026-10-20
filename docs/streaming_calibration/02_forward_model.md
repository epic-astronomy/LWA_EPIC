# 2. Forward model of the sky

The calibration solves for gains against a predicted covariance $`M(f,t)`$. Its accuracy sets the bias floor of the solution (section 3.2), so this section records the components and the choices made.

## 2.1 Components

| Component | Model | Notes |
|---|---|---|
| Cyg A | Baars et al. 1977: $`\log S = 4.695 + 0.085\log\nu - 0.178\log^2\nu`$ (ν in MHz) | 24.3 kJy at 38 MHz, 17.1 kJy at 74 MHz |
| Cas A | Baars et al. 1977 (epoch 1965) with secular decrease $`0.97 - 0.30\log(\nu/\mathrm{GHz})`$ % per year, to epoch 2026.8 | 15.4 kJy at 38 MHz, 10.1 kJy at 74 MHz. The Baars rate overestimates the low-frequency fading; de Gasperin et al. 2020 should replace it |
| Tau A | 1.6 kJy at 74 MHz, $`\nu^{-0.25}`$ | Flattening below 100 MHz |
| Vir A | 2.1 kJy at 74 MHz, $`\nu^{-0.8}`$ | VLSS scale |
| Sun | Quiet Sun, 1.5×10⁴ Jy at 74 MHz, $`\propto\nu`$ | **Flux solved** in every interval; the daytime simulation with the Sun at 3× the model recovered 2.97–3.04 |
| Diffuse sky | LFSM (Dowell et al. 2017) through pygdsm, Galactic HEALPix, evaluated at three frequency nodes per band and interpolated log-log | nside 64 (0.92°) is the minimum for the 128-grid baselines (section 3.1) |
| Beam | `lsl.sim.beam.beam_response('empirical', 'XX' or 'YY', az, alt, f)` | Evaluated once per frequency for all directions |

Point sources below 5° elevation are dropped. Above the horizon at nside 64 there are about 24,500 diffuse pixels.

## 2.2 Evaluation

```math
M = \sum_{\rm comp} A_{\rm comp}\,\mathrm{diag}(B\odot S)_{\rm comp}\,A_{\rm comp}^H,\qquad A_{as} = e^{2\pi i f\,\mathbf r_a\cdot\hat s/c}
```

Implementation choices:

- **Geometry:** full 3D stand positions (east, north, up); no w approximation enters the model.
- **Arithmetic:** geometric delays are cached per component, and the phases are evaluated as float32 sin/cos (relative error of $`M`$ 2.5×10⁻⁷). Each channel needs one complex64 GEMM.
- **Cost:** 32 solved channels × 2 polarisations take 12 s on a Xeon Silver 4210 with 8 threads, refreshed every 60 s. In 60 s the sky moves 0.25°, an eighth of the finest fringe at 75 MHz.

## 2.3 What the model does not contain

- **Unresolved and small-scale diffuse structure** below the LFSM resolution (2–5°). It is simulated as 10% multiplicative structure and costs 0.4° in the covariance solution (2.2° in the image solution).
- **Faint compact sources** (3C and AARTFAAC-catalogue sources of 100–800 Jy). These cost about 0.2°.
- **Ionospheric refraction of the A-team** (about 0.1°). It is absorbed into the direction-independent gains. The remedy is per-source position nuisance parameters (section 4.7).
- **Polarisation.** The sky is treated as unpolarised, with per-dipole power patterns.
- **RFI.** It is handled by the χ² gate rather than modelled.
- **Mutual coupling and embedded element patterns.** All stands share one beam, a second-order effect at about 5 m spacing (Jones & Wayth 2021).

On LWA-SV data the residual χ²/dof of 1.1–1.9 in clean channels (section 3.6) bounds the combined effect of these omissions at roughly the level of the thermal noise of a 0.4 s, 25 kHz snapshot.

## 2.4 Diffuse sky: the OVRO-LWA maps

LFSM has a resolution of a few degrees and 10–20% errors. LWA-SV baselines (up to about 14 λ at 38 MHz) resolve structure on 4–60° scales, so `sky.diffuse()` and `forward.build_snapshot()` take a selectable diffuse model (`model=` and `diffuse_model=` respectively). LFSM remains the default.

| `model` | Source | Resolution | Notes |
|---|---|---|---|
| `lfsm` | Dowell et al. 2017, pygdsm | a few degrees | default; A-team inpainted within 6° |
| `ovrolwa` | OVRO-LWA m-mode maps, Eastwood et al. 2018 (AJ 156, 32), hybrid with LFSM | 15–26′ | built by `research/prep_ovrolwa.py`; file from `$STREAMCAL_OVROLWA_MAP` |
| `gsm16` | GSM2016, Zheng et al. 2017, pygdsm | about 5° | control |

**The release.** LAMBDA distributes the maps at 36.528, 41.760, 46.992, … , 73.152 MHz (https://lambda.gsfc.nasa.gov/product/foreground/fg_ovrolwa_radio_maps_get.html): HEALPix nside 2048, RING, Galactic coordinates, brightness temperature in K. Compared with LFSM at 36.5 MHz:

- pixels south of dec −30° (25% of the sky) and a 2° disc around the Sun's position during the observation are exactly 0;
- the monopole and the structure that depends on declination alone (m = 0 in the equatorial frame) are missing, since m-mode analysis without autocorrelations does not constrain them. The map runs 8–16 kK below LFSM, with an offset that depends on declination;
- Cyg A, Cas A, Tau A, Vir A, Her A, Hya A, 3C 123 and 3C 353 are already removed, with residuals of 1–2 kK within 0.5° (about 1% of the Cyg A flux);
- the remaining structure agrees with LFSM to within its scale: OVRO-LWA = 1.07–1.17 × LFSM across 36.5–52.2 MHz, with correlation 0.95 per pixel at nside 256 and 10–12% rms on 5° scales.

**The hybrid map.** At each OVRO-LWA frequency the per-declination ring mean (0.5° rings, smoothed over 1.5°) is replaced by that of LFSM. South of dec −30° and in the Sun hole the map is LFSM, blended linearly over dec −30° to −26°. The A-team positions are inpainted within 1° to remove the source-subtraction residuals, which keeps the Cygnus X complex 2–3° from Cyg A. The result is stored at nside 256 (`ovrolwa_hybrid_n256.npz`, in one of the locations of `sky.OVROLWA_MAPS`). Between 36.528 and 52.224 MHz the model interpolates each pixel in log T against log ν. Below 36.528 MHz (the 28–36.5 MHz part of the EPIC band) it scales the 36.528 MHz map per pixel by the LFSM spectrum.

The model resolution need not follow the map resolution. At nside 128 every metric below agrees with nside 64 to the third decimal, because the longest LWA-SV baselines resolve about 4°. Nside 64 is therefore sufficient and costs the same as the LFSM path.

**Test on the Memo 12 captures.** The 27 night captures (01:30–08:00 UTC) were calibrated with the direction-independent `stefcal_sky` solve of `research/dd_analyze.py` at the four pilot channels, with each diffuse model at nside 64 (`research/diffuse_eval.py`; Cas A scale bound raised to 8, see below). Two measures of model error are used:

- the excess χ²/dof (χ² − 1, noise from the autocorrelations with T = 2 × nsamp);
- the cross-half model error `xerr` = Re Σ res_even res_odd* / Re Σ R_even R_odd*. The noise of the two halves is independent, so `xerr` contains model error only.

Medians over the 27 snapshots, LFSM → OVRO-LWA, baselines ≥ 1 λ:

| Channel (MHz) | Pol. | χ²/dof | Excess χ² | `xerr` | Fractional residual |
|---|---|---|---|---|---|
| 30.9 | X | 1.329 → 1.301 | −9% | 0.111 → 0.107 (−4%) | 0.253 → 0.252 |
| 30.9 | Y | 1.184 → 1.133 | −28% | 0.065 → 0.054 (−16%) | 0.220 → 0.215 |
| 35.7 | X | 1.580 → 1.500 | −14% | 0.100 → 0.093 (−7%) | 0.176 → 0.169 |
| 35.7 | Y | 1.411 → 1.298 | −28% | 0.072 → 0.061 (−16%) | 0.154 → 0.143 |
| 40.5 | X | 1.813 → 1.714 | −12% | 0.109 → 0.103 (−5%) | 0.178 → 0.173 |
| 40.5 | Y | 1.594 → 1.458 | −23% | 0.087 → 0.074 (−15%) | 0.155 → 0.143 |
| 45.3 | X | 1.880 → 1.827 | −6% | 0.127 → 0.121 (−5%) | 0.197 → 0.192 |
| 45.3 | Y | 1.578 → 1.481 | −17% | 0.090 → 0.082 (−9%) | 0.161 → 0.153 |

On longer baselines the gain grows. With umin 3 λ, `xerr` falls by 10–26% and the excess χ² by 15–42%. With umin 6 λ, `xerr` falls by 9–47% and the excess χ² by 9–48% (at 30.9 MHz Y the excess is already near zero). GSM2016 is worse than LFSM in all 24 channel, polarisation and baseline-cut combinations (excess χ² +16% to +69%, apart from 30.9 MHz Y at 6 λ, where the excess is near zero). The ordering GSM2016, LFSM, OVRO-LWA therefore follows the resolution of the map. Per snapshot, OVRO-LWA is better in almost every capture of the night.

Further results:

- **The OVRO-LWA flux scale is preferred.** A hybrid with the OVRO-LWA structure scaled by 0.9, which matches LFSM's scale, fits worse than the native scale in every channel at 1 λ.
- **A-team scales.** With OVRO-LWA the fitted Cyg A scale moves toward the catalogue (0.74–0.92 → 0.85–0.98 of Baars) and Cas A rises slightly (1.48–1.59 → 1.52–1.67). The night-long scatter (robust, 0.11–0.36) and the drift with elevation do not change systematically.
- **Guard-rail flips.** At 30.9 MHz Cas A fits at 3.5–3.8 × the Baars spectrum, against the default `stefcal_sky` bound of 4. Whether it is frozen, after which the fit fails with χ² 4–7, then varies from capture to capture with either diffuse model. `diffuse_eval.py` (`DE_SCALEMAX`) and `dd_kernelcal_real.py` (`KC_SCALEMAX`) take the bound from the environment. The de Gasperin spectra should remove the need.

**Kernel-StEFCal with each map.** The pilot of section 8.11 was rerun at nside 64 with the 8 smoothest modes, KC_LAM = 0.1 and the last quarter of the night held out (`KC_DIFFUSE`, `KC_SCALEMAX=8`). Held-out residual fraction, LFSM → OVRO-LWA (this metric includes the thermal noise, so changes are diluted):

| Channel (MHz) | Pol. | Direction-independent | With screens | Screen gain, LFSM / OVRO-LWA |
|---|---|---|---|---|
| 30.9 | X | 0.2807 → 0.2765 | 0.2666 → 0.2704 | +5.0 / +2.2% |
| 30.9 | Y | 0.2788 → 0.2680 | 0.2733 → 0.2610 | +2.0 / +2.6% |
| 35.7 | X | 0.1773 → 0.1744 | 0.1620 → 0.1591 | +8.6 / +8.8% |
| 35.7 | Y | 0.1805 → 0.1588 | 0.1632 → 0.1429 | +9.6 / +10.0% |
| 40.5 | X | 0.2661 → 0.2684 | 0.2499 → 0.2539 | +6.1 / +5.4% |
| 40.5 | Y | 0.2266 → 0.2195 | 0.2102 → 0.2058 | +7.3 / +6.3% |
| 45.3 | X | 0.2260 → 0.2222 | 0.1988 → 0.1970 | +12.0 / +11.3% |
| 45.3 | Y | 0.2398 → 0.2348 | 0.2209 → 0.2173 | +7.9 / +7.5% |

The better map lowers the held-out residual in 7 of 8 cases without screens (by up to 12%, at 35.7 MHz Y) and in 6 of 8 with them. The screens gain about as much as with LFSM, and their even/odd repeatability is unchanged (0.79–0.99). The direction-dependent component of section 8.11 is therefore not an artefact of LFSM errors, and the diffuse map does not close the gap: most of the model error lies elsewhere (`xerr` remains 5–12% of the correlated power at 1 λ).

**Remaining limits.** The X polarisation gains about half as much as Y, which points to an X-specific error that neither map contains, such as the beam at low elevation. Fitted Tau A scales of 3–5 in Y support this. The OVRO-LWA map retains extragalactic sources fainter than the eight it removes. Compact sources added to the model as point sources must therefore be subtracted from this map, or their flux is counted twice.
