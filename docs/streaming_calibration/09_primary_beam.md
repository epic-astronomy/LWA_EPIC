# 9. Primary-beam correction

## 9.1 What EPIC images contain

An EPIC image is the sky multiplied by the dipole power pattern of each polarisation:

```math
I'_{XX}(\mathbf l) \propto B_X(\mathbf l, f)\, I(\mathbf l, f),\qquad I'_{YY}(\mathbf l) \propto B_Y(\mathbf l, f)\, I(\mathbf l, f),
```

convolved with the synthesised beam. epic++ divides out only the gridding kernel's response (`gcf_correction_grid`, section 1.1). Neither epic++ nor its Python helpers apply a primary-beam correction: no code in `src/` refers to the dipole beam. Images are in instrumental units (`BUNITS = UNCALIB`).

## 9.2 How large the beam effect is for the watched sources

From the LSL `empirical` dipole model, normalised to 1 at zenith, for 2026-10-17 at LWA-SV and only while the source is above 15° elevation:

| Source | Elevation range | Stokes I factor, 38 MHz | Change over the day, 38 MHz | XX/YY, 38 MHz | Change over the day, 74 MHz |
|---|---|---|---|---|---|
| Crab | 16–78° | 0.13–0.96 | ×7.5 | 0.97–1.64 | ×5.8 |
| PSR B0809+74 | 19–50° | 0.16–0.62 | ×3.9 | 0.57–0.76 | ×3.1 |
| PSR B1133+16 | 17–71° | 0.14–0.90 | ×6.6 | 0.94–1.71 | ×5.1 |
| Cyg A | 16–83° | 0.13–0.99 | ×7.8 | 1.00–1.25 | ×6.2 |
| Sun | 15–46° | 0.12–0.56 | ×4.8 | 0.73–1.41 | ×3.7 |

A watched source's apparent brightness therefore changes by a factor of 3–8 over a day purely through the beam. The X and Y dipoles see it differently by up to 1.7×, and the ratio depends on azimuth.

## 9.3 Is it required, and will it help?

**Detection: no.** Dividing a pixel by $`B(\mathbf l)`$ scales its signal and its noise equally. The S/N of a transient or a pulse is unchanged, so detection thresholds should stay in S/N units, which the FRB and pulsar searches use.

**Flux densities, fluences and light curves: yes.** Without correction:

- **Light curves:** a pulsar or the Crab followed through a day varies by ×3–8 from the beam alone. Long-term pulsar monitoring (section 2 of `pulsars_and_calibration.md`) needs the correction to compare epochs.
- **Fluence limits and fluences:** these are biased low by $`B`$, and the bias depends on elevation. The FRB fluence limits in the notes assume zenith.
- **Spectra:** the beam's frequency dependence across 16–88 MHz enters measured spectral indices.
- **Stokes I:** forming I as XX+YY weights X and Y unequally where $`B_X\ne B_Y`$, up to 1.7× here, and mixes polarised emission into I.

**Image quality: no.** A primary-beam correction does not change calibration, sidelobes or artefacts. The calibration already includes the beam in its sky model (sections 2 and 6).

## 9.4 How to apply it

The correction belongs downstream of the GPU, applied per image, channel and polarisation:

```math
I_{\rm Jy}(\mathbf l) = \frac{1}{2}\left[\frac{I'_{XX}(\mathbf l)}{s_X\,B_X(\mathbf l)} + \frac{I'_{YY}(\mathbf l)}{s_Y\,B_Y(\mathbf l)}\right],
\qquad B > B_{\min}\ (\text{elevation} \gtrsim 15^\circ),
```

where $`s_{X,Y}`$ are the absolute flux scales the calibration service solves (counts per Jy, kept by `daemon.table()`, section 4.3).

Where it fits in the pipeline:

- **Pixel extractor:** multiply the extracted values of a watched source by $`1/(s\,B)`$ at its exact position (`source_pix`).
- **FITS writer:** store per-channel beam maps, or a beam-corrected Stokes I plane, with a header keyword for the flux scale.

The cost is negligible. The beam changes only with frequency in the station frame, so the maps are fixed per channel.

Below about 15° elevation $`B`$ falls to 0.1 and the correction amplifies noise and model errors, so pixels there should be masked or carry the beam as a weight.

## 9.5 Accuracy is limited by the beam model

The real data show the limit. With the LSL beam in the model, the fitted Cyg A flux (section 8.5) moved from 0.6–0.8 of the catalogue near zenith (04:00 UTC) to 1.1–1.3 two hours later (06:00 UTC). Part of that is the catalogue spectrum, but the change with elevation points to beam-model errors of tens of per cent.

The calibration service already solves an apparent flux scale for every bright source in every 10 s table (section 6.2). Those scales are samples of $`B_{\rm true}/B_{\rm model}`$ along the A-team tracks, about 360 per hour per source per channel. Accumulated over days, they give an empirical correction to the beam model in azimuth and elevation, the method used to bootstrap LWA and OVRO-LWA beams from A-team transits. The A-team catalogue spectra (de Gasperin et al. 2020) must be fixed first, or the catalogue error enters the beam.

## 9.6 Recommendation

1. Add the per-channel X and Y beam maps and the flux scales to the image metadata.
2. Apply the correction in the pixel extractor for watched sources, so that light curves and fluences are in Jy.
3. Keep the search statistics in S/N units.
4. Log the per-source flux scales from streamcal, and refine the beam model from them once the A-team spectra are updated.

## 9.7 Beam correction from the TBF captures

The A-team flux scales of section 2.4 (Cas A about 0.83, Cyg A about 0.90, drifting with elevation), the weaker response of X than Y to better diffuse maps, and Tau A fits of 3–5 in Y at low elevation all point to errors in the LSL `empirical` dipole model, which multiplies every model component. A smooth multiplicative correction to that model was therefore fitted to the 27 night captures of 2023-09-17 (01:30–08:00 UTC; ADP channels 1236, 1428, 1620 and 1812 at 30.9, 35.7, 40.5 and 45.3 MHz) with `research/beam_fit.py`.

**Model.** Per channel and polarisation,

```math
B_{\rm true}(\phi, \theta, f) = B_{\rm LSL}(\phi, \theta, f)\, C(\phi, \theta;\, \mathbf p),\qquad
\log C = \sum_j p_j Y_j(\phi, \theta),
```

with $`\rho = \theta / 90^\circ`$ ($`\theta`$ the zenith angle) and $`\phi`$ the azimuth from north through east (`streamcal/beamcorr.py`):

| Term | $`Y_j`$ | Effect |
|---|---|---|
| r2 | $`2\rho^2`$ | beam width (Zernike $`Z_2^0`$ less its zenith value) |
| r4 | $`6\rho^4 - 6\rho^2`$ | mid-elevation shape, zero at the zenith and the horizon ($`Z_4^0 - 1`$) |
| c1, s1 | $`\rho\cos\phi`$, $`\rho\sin\phi`$ | north–south and east–west gradient (tilt) |
| c2, s2 | $`\rho^2\cos 2\phi`$, $`\rho^2\sin 2\phi`$ | north–south against east–west width, and the diagonal |

Every term vanishes at the zenith, which fixes the gauge: $`C = 1`$ at the zenith, so an overall constant remains with the gains and the source flux scales, with which it is exactly degenerate. The terms are low-order Zernike polynomials on the disc of the visible hemisphere, all of order unity, so the normal matrix is well conditioned and the fitted coefficients are directly comparable. The exponential keeps $`C`$ positive.

**Unknowns and solver.**

- Per snapshot: direction-independent gains (StEFCal), and A-team position offsets (refraction) taken from a `stefcal_sky` pre-fit against the uncorrected model and then held.
- Shared by all training snapshots: $`\mathbf p`$, and one flux scale per A-team source. A per-snapshot flux would be degenerate with the beam toward that source.
- Tau A and Vir A are seen only below 25° on short tracks. Fitted freely, their scales run to 12 and 0.05 while $`\mathbf p`$ is unchanged to the third decimal. They are tied to the mean of the Cas A and Cyg A scales.

The model visibilities are $`M(\mathbf p, \mathbf s) = \sum_{\rm pix} C\,S\,\mathbf a\mathbf a^H + \sum_k s_k C_k S_k \mathbf a_k \mathbf a_k^H`$. Their derivatives with respect to $`\mathbf p`$ are analytic (one batched GEMM per snapshot). $`(\mathbf p, \mathbf s)`$ are fitted by Levenberg–Marquardt on normal equations pooled over the training snapshots, with the per-snapshot gains profiled out of each step: the Schur complement of the gain block, which has a closed form for gains perturbed as $`g_a(1 + u_a + i v_a)`$. StEFCal re-solves the gains after each step. A plain alternation between gains and beam converges only slowly, because the gains absorb part of each change in $`\mathbf p`$. With the projection, three to five steps suffice. Weights follow `stefcal`: baselines of at least 1 λ, with autocorrelations and flagged stands excluded. Each snapshot is weighted by $`1/\sum|R|^2W`$, so the cost is the sum of fractional residuals. The diffuse model is the OVRO-LWA hybrid (section 2.4) at nside 32. At nside 64 the coefficients agree to 0.003.

**Validation design.** The fit uses the first 21 snapshots (01:30–06:30 UTC). The last 6 (06:45–08:00 UTC, sky rotated) are held out. Each snapshot is scored with the uncorrected and the corrected beam in two ways:

- `op`: the operational `stefcal_sky` solve, with free per-snapshot source fluxes and offsets (the `sky_eval.py` scoring);
- `fixed`: StEFCal only, with the source scales shared from training (re-fitted with $`\mathbf p = 0`$ for the uncorrected case).

The metrics are χ²/dof (noise from the autocorrelations), the fractional residual, and the cross-half model error `xerr` (section 2.4), which is free of noise.

**Simulation.** Data were simulated from the OVRO-LWA sky with random gains, thermal noise from the measured autocorrelations, Cas A and Cyg A at 0.85 and 0.90, Tau A and Vir A at 1.0 (in conflict with the tie), and an injected correction $`\mathbf p = (-0.25, 0.15, 0.12, -0.08, 0.15, 0.05)`$ ($`C`$ = 0.51–0.72 at 20°, 0.95–0.99 at 75°). In all 8 channel–polarisation pairs the fit recovers $`\log C`$ to 0.008–0.018 rms above 20° elevation. The recovered Cas A and Cyg A scales are 0.86 and 0.91, and χ²/dof falls to 0.99 with `xerr` at zero. The even and odd halves agree to 0.001–0.007. When the same data are fitted with the LFSM map (a deliberate diffuse-model error), the azimuthal terms survive. The elevation terms do not: $`\log C`$ is off by 0.08–0.11 rms, $`C`$ at 20° is 7–19% low, and the A-team scales absorb the difference (0.75 against 0.85). On baselines of at least 1 λ, the elevation dependence of the beam is partly degenerate with the large-scale brightness of the diffuse map.

**Fitted correction** ($`C`$ at elevation 30°, 45° and 60° toward north, east, south and west; OVRO-LWA map; both halves of the data):

| MHz | Pol. | 30°: N E S W | 45°: N E S W | 60°: N E S W | p (r2, r4, c1, s1, c2, s2) |
|---|---|---|---|---|---|
| 30.9 | X | 1.16 0.88 1.32 0.85 | 1.07 0.93 1.18 0.91 | 1.02 0.97 1.09 0.95 | +0.06 +0.01 −0.10 +0.03 +0.40 −0.00 |
| 35.7 | X | 1.14 0.75 1.21 0.77 | 1.07 0.84 1.12 0.87 | 1.02 0.92 1.05 0.94 | −0.05 +0.01 −0.04 −0.02 +0.49 +0.05 |
| 40.5 | X | 1.21 0.66 1.25 0.76 | 1.11 0.78 1.13 0.86 | 1.04 0.88 1.06 0.95 | −0.07 +0.01 −0.02 −0.11 +0.63 +0.01 |
| 45.3 | X | 1.33 0.71 1.42 0.83 | 1.19 0.83 1.25 0.93 | 1.08 0.91 1.12 0.99 | −0.09 −0.07 −0.05 −0.12 +0.66 −0.08 |
| 30.9 | Y | 0.85 1.28 1.03 1.18 | 0.90 1.16 1.04 1.09 | 0.94 1.08 1.03 1.04 | +0.06 −0.01 −0.14 +0.06 −0.31 −0.12 |
| 35.7 | Y | 0.73 1.28 0.93 1.06 | 0.82 1.17 0.98 1.02 | 0.90 1.09 1.01 0.99 | −0.03 −0.01 −0.18 +0.14 −0.39 −0.03 |
| 40.5 | Y | 0.68 1.35 0.94 1.12 | 0.79 1.22 1.01 1.06 | 0.88 1.11 1.03 1.01 | −0.08 −0.04 −0.24 +0.14 −0.48 −0.02 |
| 45.3 | Y | 0.72 1.48 1.02 1.19 | 0.82 1.30 1.07 1.11 | 0.89 1.15 1.07 1.03 | −0.06 −0.08 −0.27 +0.16 −0.50 −0.04 |

The dominant term is c2, with opposite signs in X and Y. In the LSL model, XX at 20° elevation is 0.44–0.61 of YY toward north and south and 1.65–2.23 toward east and west. The correction moves the two patterns toward each other: the fitted $`\log C`$ correlates with $`\log(B_{\rm other}/B_{\rm self})`$ at 0.90–0.93 in X and 0.65–0.73 in Y. In the c2 coefficient of $`\log(B_X/B_Y)`$, the corrected asymmetry is about 50% of the LSL value at 30.9 MHz and 20% at 35.7 MHz. At 40.5–45.3 MHz it is near zero, slightly reversed (−6% and −16%) and therefore uncertain. The polarisation labels are not swapped. With each polarisation modelled by the other dipole's pattern (`BF_SWAP=1`), the uncorrected fit is worse (held-out χ²/dof 1.472 against 1.415), and so is the corrected one (1.434 against 1.326). Y also has a gradient (c1 −0.14 to −0.27, s1 +0.06 to +0.16) toward the south-east, about 150° in azimuth. The ground plane of the stands slopes 1.8° with its normal toward azimuth 125°, which agrees in sense, but X has no matching gradient, so the slope does not explain the term alone.

**Results on the held-out snapshots** (medians over the 6 test snapshots, uncorrected → corrected):

| MHz | Pol. | `op` χ²/dof | `op` residual | `op` xerr | `fixed` residual | `fixed` xerr |
|---|---|---|---|---|---|---|
| 30.9 | X | 1.244 → 1.213 | 0.298 → 0.292 | 0.103 → 0.095 | 0.353 → 0.352 | 0.162 → 0.160 |
| 35.7 | X | 1.409 → 1.326 | 0.168 → 0.160 | 0.084 → 0.076 | 0.241 → 0.226 | 0.161 → 0.132 |
| 40.5 | X | 1.687 → 1.565 | 0.172 → 0.162 | 0.103 → 0.092 | 0.232 → 0.224 | 0.168 → 0.159 |
| 45.3 | X | 1.727 → 1.602 | 0.186 → 0.175 | 0.112 → 0.102 | 0.222 → 0.205 | 0.158 → 0.138 |
| 30.9 | Y | 1.130 → 1.089 | 0.279 → 0.271 | 0.063 → 0.052 | 0.329 → 0.325 | 0.108 → 0.103 |
| 35.7 | Y | 1.272 → 1.198 | 0.162 → 0.154 | 0.063 → 0.055 | 0.220 → 0.212 | 0.131 → 0.122 |
| 40.5 | Y | 1.415 → 1.299 | 0.139 → 0.129 | 0.068 → 0.056 | 0.209 → 0.196 | 0.138 → 0.125 |
| 45.3 | Y | 1.446 → 1.338 | 0.153 → 0.149 | 0.083 → 0.074 | 0.190 → 0.178 | 0.118 → 0.109 |
| all | | 1.415 → 1.326 | 0.173 → 0.162 | 0.085 → 0.073 | 0.239 → 0.230 | 0.152 → 0.129 |

On the held-out snapshots the correction lowers `xerr` in 8 of 8 pairs in both scorings, and χ²/dof in 8 of 8 with `op`. In the median, the operational solve loses 21% of its excess χ² (0.415 → 0.326), 6.5% of its residual and 14% of its model error. On the training snapshots the gains are the same (χ²/dof 1.485 → 1.402, `xerr` 0.088 → 0.079), so the correction does not over-fit. Variants, as held-out `op` χ²/dof (1.415 uncorrected):

- the radial terms alone (r2, r4): 1.432, no gain;
- r2, r4, c2 and s2: 1.339, most of the gain;
- all six terms: 1.326;
- with the LFSM map instead: 1.525 → 1.371, so the beam correction is worth more than the change of map. c2 (0.34–0.61 in X, −0.39 to −0.55 in Y) agrees with the OVRO-LWA fit to within 0.05–0.09, about 15% of its value. r2 and r4 differ by up to 0.07, which is comparable to their values.

**A-team scales.**

- **Shared scales (training).** These move toward 1, from Cas A 0.77–0.87 and Cyg A 0.84–0.99 to 0.82–0.91 and 0.87–1.02. The means rise by 3% and 1.6%.
- **Drift with elevation (per-snapshot `op` scales).** In X the drift shrinks, in slope per 10° and robust scatter:
  - Cas A: slope +0.10 to +0.27 → −0.02 to +0.11; scatter 0.13–0.19 → 0.08–0.14;
  - Cyg A: slope −0.04 to −0.09 → −0.02 to −0.03.
- **Drift in Y.** Y had little drift and keeps it, apart from Cas A at 40.5–45.3 MHz (+0.11 → +0.14 to +0.16).
- **Cyg A scatter.** Its per-snapshot scatter (0.15–0.28) does not change, so it has another cause.
- **Tau A** (held-out, 12–24° elevation, when not frozen by the guard rails). Before the correction it fits 3.0–4.3 in Y and 0.44–0.76 in X. After it, X gives 1.0–2.1 (one value of 3.4 at 9°) and Y 1.4–2.9 (at 35.7 MHz: 0.44–0.76 → 1.5–1.9 in X, 3.0–3.3 → 1.44–1.51 in Y). The X/Y discrepancy disappears. The common level of 1.5–2, about twice the Cas A and Cyg A scale, remains at an elevation where $`C`$ is an extrapolation.
- **Vir A** (6–12°) is frozen in every capture.

**X/Y asymmetry.** The excess χ² of X stays 1.5–2.4 times that of Y, and the `xerr` ratio stays 1.5–2.0. Both polarisations improve by similar fractions. The asymmetry of the beam model shrinks, but the X-specific error of section 2.4 is not a smooth beam error of this form.

**Consistency.** The even and odd halves agree to 0.001–0.009 rms in $`\log C`$ above 20°. Across channels, the patterns correlate at 0.73–0.99 (0.98–0.99 between 35.7 and 40.5 MHz), with an rms difference of 0.035–0.115. The main terms change smoothly with frequency: c2 runs +0.40, +0.49, +0.63, +0.66 in X and −0.31, −0.39, −0.48, −0.50 in Y, and c1 in Y runs −0.14 to −0.27.

**Kernel-StEFCal with the corrected beam.** The pilot of section 8.11 (8 smoothest modes, KC_LAM 0.1, last quarter held out, OVRO-LWA map at nside 64, `KC_SCALEMAX=8`) was rerun with `KC_BEAM` set to the fitted table:

| MHz | Pol. | Direction-independent, LSL → corrected | With screens, LSL → corrected | Screen gain, LSL / corrected |
|---|---|---|---|---|
| 30.9 | X | 0.2778 → 0.2731 | 0.2717 → 0.2720 | +2.2 / +0.4% |
| 30.9 | Y | 0.2693 → 0.2613 | 0.2631 → 0.2538 | +2.3 / +2.8% |
| 35.7 | X | 0.1744 → 0.1654 | 0.1591 → 0.1525 | +8.7 / +7.8% |
| 35.7 | Y | 0.1589 → 0.1515 | 0.1431 → 0.1368 | +9.9 / +9.7% |
| 40.5 | X | 0.1664 → 0.1572 | 0.1536 → 0.1464 | +7.7 / +6.9% |
| 40.5 | Y | 0.1243 → 0.1158 | 0.1108 → 0.1033 | +10.9 / +10.8% |
| 45.3 | X | 0.2222 → 0.2121 | 0.1970 → 0.1878 | +11.3 / +11.5% |
| 45.3 | Y | 0.1528 → 0.1458 | 0.1356 → 0.1284 | +11.3 / +11.9% |

The beam correction lowers the held-out direction-independent residual by 2–7% in every pair. The screen gain is unchanged within about 1% except at 30.9 MHz X, and the screens' even/odd repeatability is unchanged. The common beam and the per-stand screens are therefore separate effects: the screen gauge removes any pattern common to all stands, and the two corrections add.

**Conclusions.**

1. A six-term correction to the LSL `empirical` beam improves direction-independent calibration on held-out, sky-rotated data in every channel and polarisation: χ²/dof falls from 1.415 to 1.326, the excess χ² by 21% and the model error by 14%. The gain is larger than that from replacing LFSM with the OVRO-LWA map.
2. The robust part is azimuthal: the LSL model overstates the difference between the X and Y patterns, and overstates it more at higher frequency. The elevation dependence is weakly determined and depends on the diffuse map.
3. The correction can enter the forward model through `build_snapshot(beam_corr=beamcorr.Table(path))`. It is not enabled by default in the daemon.

**Caveats.**

- **Coverage.** One night, one 15 MHz band and 21 training snapshots. The held-out quarter (1.25 h) differs in sky but not in season.
- **Direct constraints.** Only Cas A (34–65° elevation, northern sky) and Cyg A (36–83°) constrain the beam directly. Elsewhere the constraint is the diffuse map, so map errors on large scales enter the elevation terms.
- **Low elevation.** Below about 25° $`C`$ extrapolates a polynomial. The 20° values (for example 1.31–1.43 and 0.57–0.61 in X at 40–45 MHz) should not be used for flux correction.
- **Absolute level.** The gauge $`C = 1`$ at the zenith leaves the absolute zenith response, and its frequency dependence, to the gains and the A-team scales.
- **Model scope.** XX and YY are fitted as independent scalar patterns. Polarisation leakage, mutual coupling between stands and per-stand pattern differences are outside the model. The per-stand differences are the screens of section 8.
- **Tau A** still fits at 1.5–2 times the common scale below 25°.

**What more data would add.**

- **More nights and seasons.** Cas A and Cyg A at other hour angles, and Tau A and Vir A at high elevation (Tau A transits at 78° on winter nights), give direct constraints over the whole sky. They would separate the beam from the diffuse map and test whether the correction is stable between nights and ionospheric conditions.
- **More sources.** Bright 3C and VLSSr sources solved for flux along their tracks, and the Sun by day (with its variability as a free parameter), sample more directions.
- **Other bands.** The trend of c2 with frequency (31–45 MHz) needs the 16–30 and 50–88 MHz bands to establish a model across the EPIC range, and to compare with the frequencies at which the LSL model was derived.
- **Full polarisation.** XY correlations from the tap would constrain the leakage terms that a scalar correction cannot represent.
- **Accumulated daemon solutions** (section 9.5). The per-source scales solved every 10 s over days give the same constraint continuously.

Code: `streamcal/beamcorr.py` (basis, table, `build_snapshot` hook), `research/beam_fit.py` (fit and hold-out scoring; `BF_SIM` for the simulation, `BF_SWAP` for the label test), `research/beam_fit_report.py` (tables), and `KC_BEAM` in `research/dd_kernelcal_real.py`. Outputs are in epic1:/data/streamcal/beam/.
