# 8. Direction-dependent gains through per-antenna kernels: methodology

The streaming calibration of sections 1–7 is tagged `streamcal-v1.0` (29c5cc0). It solves direction-independent gains per stand, and it carries the direction-dependent behaviour of the ionosphere in the sky model. This section asks whether per-stand direction-dependent gains, applied through the gridding kernels, would improve EPIC images, and how such gains would be measured. No implementation is proposed until the answer is known.

## 8.1 The kernel is a per-antenna direction-dependent weight

With the epic++ gridder, the E-field image of one sequence is

```math
e(\mathbf l) = \sum_a w_a\,x_a\;\hat K_a(\mathbf l)\,e^{2\pi i\,\mathbf u_a\cdot\mathbf l},
\qquad
\hat K_a(\mathbf l) = \sum_{\rm cells} K_a(\mathbf c)\,e^{-2\pi i\,\delta\,(\mathbf c-\mathbf p_a)\cdot\mathbf l}.
```

Here $`\mathbf p_a`$ is the stand's position on the grid in cells, and $`\delta = 0.424\lambda`$ per cell. The transform of the kernel, $`\hat K_a`$, multiplies stand $`a`$'s contribution in every direction. It is a direction-dependent complex weight per stand. A per-stand direction-dependent gain $`J_a(\mathbf l)`$ is corrected by

```math
\hat K'_a(\mathbf l) = \hat K_a(\mathbf l)\,c_a(\mathbf l),\qquad c_a = J_a^{-1}\ \text{or}\ J_a^*/|J_a|^2 .
```

This is A-projection (Bhatnagar et al. 2008) applied to E-fields. epic++ already uploads complex per-stand, per-channel kernels for w-projection (`m_gcf_elem_cplx`). A direction-dependent correction folds into the same numbers, with no change to the GPU kernel and no extra cost, provided the support stays at 5×5. Shared memory is exactly full at support 5, so larger kernels are not free. The kernels are shared by the X and Y dipoles of a stand, so polarisation-specific corrections would need a second set. The gridding correction and the autocorrelation removal are computed from the actual kernels, so both follow automatically.

## 8.2 What a 5×5 kernel can represent

With support $`S`$, $`\hat K'_a`$ is a two-dimensional trigonometric polynomial of degree $`(S-1)/2`$ in $`\delta\,\mathbf l`$. It can follow only smooth structure across the sky.

`research/dd_kernel_fit.py` fits the best $`S\times S`$ kernel to the production kernel response multiplied by random smooth correction screens. The screens have rms 0.1 in phase and log-amplitude and the stated correlation length. The fit is least squares over the visible sky, weighted by sin(elevation). The table gives the median over stands of the fraction of the correction captured:

| Screen scale | S = 5 (production) | 7 | 9 | 11 |
|---|---|---|---|---|
| 5° | 0.24–0.28 | 0.41–0.45 | 0.58 | 0.71–0.73 |
| 10° | 0.64–0.69 | 0.85–0.86 | 0.94–0.95 | 0.98 |
| 20° | 0.94–0.95 | 0.99 | 0.99 | 1.00 |
| 40° | 0.99 | 1.00 | 1.00 | 1.00 |

The ranges span 34.5 and 75.8 MHz; the representable scale in direction cosines does not depend on frequency.

The production kernels can therefore carry direction-dependent corrections smoother than about 15–20°. They cannot carry the fine ripple that mutual coupling imposes on embedded element patterns, whose period is a few degrees (λ divided by the station size).

## 8.3 What an uncorrected direction-dependent error costs

Per-stand gain errors toward a source of flux $`S`$, with rms $`\sigma`$ relative to the direction-independent solution, reduce the source's peak only by about $`\sigma^2`$. They also add a static error pattern, the first-order term $`2\,\mathrm{Re}[\psi^*(\mathbf l)\sum_a\varepsilon_a\psi_a(\mathbf l)]`$, with rms of about $`S\,\sigma/\sqrt{N}`$.

`research/dd_image_impact.py` measures it with the exact image operator, for Cyg A at 06:00 UTC on 2026-10-16, against the thermal noise of one 83.6 ms single-channel image of the full forward-modelled sky:

| Error toward Cyg A | Artefact rms, 34.5 / 75.8 MHz | Fraction of one image's noise (161 / 143 Jy) | Integration at which artefact equals noise |
|---|---|---|---|
| 1° | 1.7 / 1.4 Jy | 1% | about 13 min |
| 3° | 5.1 / 4.5 Jy | 3% | about 85 s |
| 10° | 17 / 15 Jy | 10% | about 7.5 s |

The artefact peaks near the source, at about 10× its rms. Smooth direction-dependent errors give artefacts that add nearly coherently across channels, while band-summing reduces the noise by up to $`\sqrt{256}`$. Errors of a few degrees therefore matter for any integrated or band-summed product near the A-team. They do not matter for single-channel snapshots.

## 8.4 Measuring per-antenna direction-dependent gains

**Directional solutions.** Per channel and polarisation:

1. The direction-independent solve (`stefcal_sky`) gives gains, source fluxes and positions.
2. Gains toward each bright source are fitted to the data with the diffuse model and the other sources (with their current directional gains) subtracted. This alternates over sources four times, in the manner of SAGE.
3. The statistic is the ratio of the gains toward two sources, $`d_a = g_a(\text{Cyg A})/g_a(\text{Cas A})`$, with the common phase, the tip/tilt (refraction) and the median amplitude removed.

**Separating signal from noise.** The samples are split into even and odd halves, which are independent realisations of the same sky and instrument:

- **Signal** variance is $`E[d^{\rm even} d^{\rm odd}]`$, reported signed with its standard error.
- **Noise** variance is $`E[(d^{\rm even}-d^{\rm odd})^2]/2`$.
- **Smoothness:** the correlation of $`d`$ between neighbouring channels identifies effects that are smooth in frequency, as instrumental effects are; noise gives zero.

**Validation on simulated captures** (`research/dd_simcheck.py`). The setup matches the TBF captures: 16 channels at 28–47 MHz, 12,000 samples per half, truth sky different from the model.

| Simulation | Band-averaged phase variance (X / Y) | Amplitude variance (X / Y) |
|---|---|---|
| No direction-dependent gains | 1.1 / 1.6 deg² (floor from sky-model error) | 2.6 / 3.5 %² |
| Injected 3° (9 deg²) and 5% (25 %²) | 8.8 / 10.9 deg² | 32 / 27 %² |

The method recovers injected effects. Its floor, set by model error, is about 1.2° and 1.7%.

## 8.5 Real data: what the first measurement found

The first application to LWA-SV TBF captures (EPIC Memo 12, 2023-09-17) exposed two problems in the forward model and one in the solver. None of them were visible in simulation, because there truth and model share the same sky:

1. **A-team counted twice.** The LFSM map contains Cyg A and Cas A, smoothed to its resolution (about 12 kJy above the surroundings within 2° at 38 MHz), and the model added them again as point sources. The fitted Cyg A flux fell to about 0.3 of the catalogue and appeared to decorrelate on longer baselines. `sky.remove_compact` inpaints the A-team from the map (r < 6°, from the 6–9° annulus). With the map fixed, Cyg A comes out at 0.6–1.3 and Cas A at 1.1–1.9 of the catalogue, independent of the baseline cutoff.
2. **Cas A too faint.** Cas A is about 1.5× brighter than the Baars spectrum with its secular decrease, consistent with the known overestimate of the fading at low frequencies. The de Gasperin et al. (2020) spectra should replace it (section 4.7).
3. **Unguarded sky parameters.** Free source positions ran away (6–11°) and fluxes collapsed. `stefcal_sky` guards them:
   - offsets are limited to 0.5° × (74 MHz/f)² and scales to 0.25–4 (Sun 0.1–1000);
   - a source that leaves those bounds is reset to the catalogue and frozen for the rest of the solve;
   - a cold start solves the gains against the nominal sky first.

   On the real data the guarded solver lowers χ²/dof from 2.47 to 1.73 at 38 MHz, with offsets of 0.3–0.5°.

With the corrected model, Cyg A versus Cas A shows a direction-dependent signal well above the noise:

| Capture (UTC) | Pol. | Phase variance per channel | Amplitude variance | Noise per channel | Neighbour-channel correlation |
|---|---|---|---|---|---|
| 02:00 | X | 198 ± 54 deg² | 175 ± 23 %² | 33°, 25% | 0.59 |
| 04:00 | X / Y | 186 ± 29 / 206 ± 31 deg² | 90 ± 7 / 64 ± 11 %² | 10°, 16% | 0.65 / 0.70 |
| 06:00 | X / Y | 281 ± 48 / 353 ± 48 deg² | 63 ± 6 / 39 ± 11 %² | 8–12°, 14% | 0.80 / 0.71 |

That is 14–19° in phase and 6–9% in amplitude, smooth in frequency. Averaged over the band, the phase variance is smaller than per channel, so the signal varies across the band but smoothly. This is the signature of a phase proportional to frequency, $`2\pi f\,\Delta\mathbf r_a\cdot(\hat s_{\rm CygA}-\hat s_{\rm CasA})/c`$. It is what stand position errors, or offsets of each dipole's electrical phase centre, would produce. Section 8.6 tests this.

## 8.6 What the direction-dependent phase is

Twelve further captures from the same night extend the test to 15 epochs, 01:00–08:00 UTC at 30-minute intervals (`research/dd_diagnose.py`, `dd_positions2.py`, `dd_freqscale.py`).

| Test | Result | Reading |
|---|---|---|
| X/Y correlation of the band-averaged phase | 0.7–0.93 in 10 of 15 epochs | shared by the two dipoles of a stand |
| Correlation with the capture 30 min earlier | 0.5–0.98 (0.93–0.98 from 05:00 to 06:00) | persistent over hours |
| Short baselines excluded (≥ 6λ) | phase variance unchanged (e.g. 272 → 267 deg²) | not diffuse-model leakage |
| Low-band against high-band phase, cross-half slope | +0.58 to +0.69 (03:00–05:00) | ionosphere predicts 1.48: **not ionospheric** |
| Fixed position offset per stand, cross-validated | −1 to −3% of the variance | **not a position error** |
| Phase-centre offset with a free offset per capture, cross-validated | −2 to −3% | **not a phase-centre height** |
| Free delay per stand and capture | 12–13% beyond a constant offset | mostly frequency-independent |

The effect is therefore a per-stand, mostly frequency-independent phase toward one direction relative to another, about 15–20° rms between Cyg A and Cas A. It is shared by X and Y and stable over hours. Its properties fit differences in the embedded element patterns of the stands, from the ground screen, local terrain and mutual coupling. They do not fit propagation or geometry.

**An independent dataset confirms it.** The 20-minute TBN capture at 40.0 MHz on 2021-04-04 (pre-NDP) was correlated in 10 s chunks, split into halves (`research/tbn_cache.py`, then `tbn_tcc.py` on the GPUs; `tbn_dd_persist.py`):

| | X | Y |
|---|---|---|
| Per-chunk direction-dependent phase signal | 19.9° rms (noise 2.8°) | 23.0° rms (noise 3.6°) |
| Correlation between chunks 15–60 s apart | 0.99 | 0.98 |
| Correlation between chunks 1–3 min apart | 0.99 | 0.97 |
| Correlation between chunks 3–8 min apart | 0.97 | 0.90 |
| Correlation between chunks 8–20 min apart | 0.91 | 0.73 |
| X/Y correlation | +0.47 | |

The table covers 115 chunks of 10 s each. The GPU correlation took 7 s per chunk, mostly reordering in Python; the CPU had taken about 450 s.

## 8.7 Why per-antenna kernels cannot apply the correction in a direct imager

A kernel correction multiplies stand $`a`$'s contribution by $`c_a(\mathbf p)`$ as a function of the **image pixel** $`\mathbf p`$. A direction-dependent gain belongs to the **source direction** $`\hat s`$:

```math
e(\mathbf p) = \sum_a w_a\, J_a(\hat s)\, c_a(\mathbf p)\, \hat K_a(\mathbf p)\, e^{2\pi i\,\mathbf u_a\cdot(\hat s-\mathbf p)} \;+\;\ldots
```

The correction cancels $`J_a(\hat s)`$ only where $`\mathbf p\approx\hat s`$, on the source's main lobe. Its sidelobes receive the correction for the wrong direction. Their error becomes $`\varepsilon_a(\hat s)-\varepsilon_a(\mathbf p)`$, larger than the uncorrected $`\varepsilon_a(\hat s)`$ wherever the two directions are not correlated. This is the known property of A-projection (Bhatnagar et al. 2008): it is exact only inside major cycles that subtract a model and degrid with the same direction-dependent terms. EPIC forms dirty images with no deconvolution, so kernel corrections help compact-source photometry and harm sidelobe fidelity.

The tests (`research/dd_kernel_unit.py`, `dd_gain_sim.py`, `dd_kernelcal_sim.py`) confirm each part:

- **Exact for direction-independent gains.** Kernels $`K_a/g_a`$ reproduce the ideal image to $`5\times10^{-9}`$.
- **Point sources.** A 10° linear screen is applied correctly at the source direction (to 0.6–0.8°). The peak of a single point source recovers 70% of its loss (46,579 → 46,767, ideal 46,847).
- **The whole sky gets worse.** The image error over the visible sky rises from 0.025 to 0.037 of the image norm. The 5×5 fits reproduce their targets to 0.6% everywhere on the disk, so the cause is the main-lobe/sidelobe mismatch, not representability.
- **Even perfect screens make it worse.** At 34.5 MHz on 2021-04-04 (held-out LST), the error over the sky is 151 Jy with direction-independent calibration, 295 Jy with kernels from the solved screens, and 307 Jy with kernels from the true screens.
- **Two-point bumps around Cyg A and Cas A** (`dd_gain_sim.py`) do not help either. With the screens applied to the sources only, kernels gain at most 15%, while image-domain subtraction of the two sources with their measured gains reduces the error 3.5–6×. With the screens applied to the diffuse sky, the dominant error at 34.5 MHz (88–807 Jy, against a single-channel thermal noise of 161 Jy) is untouched by either.

## 8.8 Estimating whole-sky per-stand screens: kernel-StEFCal

The screens can be estimated even though kernels cannot apply them. Each stand's response is parameterised as $`J_a(\hat s) = \sum_j \beta_{aj}\,\phi_j(\hat s)`$ over the 25 Fourier modes $`\phi_j = e^{2\pi i\,\delta\,\mathbf c_j\cdot\mathbf l}`$ ($`\mathbf c_j\in\{-2..2\}^2`$) that a 5×5 kernel spans; $`\beta_{a0}`$ is the direction-independent gain. The covariance is then bilinear:

```math
R_{ab} = \sum_{j} \beta_{aj}\,X_{ab,j},\qquad X_a[j,b] = \sum_s \phi_j(\hat s)\,a_a(\hat s)\,S(\hat s)\,\overline{a_b(\hat s)\,J_b(\hat s)}.
```

Each stand is solved by linear least squares: 25 unknowns, about 228 baselines per snapshot and channel. The solve alternates over stands with StEFCal-style averaging and a Tikhonov pull toward the previous iterate (`research/dd_kernelcal_sim.py`). The whole diffuse sky constrains the screens, not two points. Pooling snapshots across LST separates the instrument, which is fixed in the station frame, from sky-model errors, which rotate with the sky.

In simulation the test used:

- **Truth:** screens with 20° correlation length scaled to the measured 20° Cyg A/Cas A phase rms; truth sky LFSM nside 64 × (1 + 10% structure) + 25 faint sources.
- **Model:** LFSM nside 32.
- **Fit:** 5 snapshots over 6 h of LST.

It recovered the screens at the source directions to 0.6° and 1.9°.

The screens are useful for **predicting** the EPIC image, not for correcting it. The held-out image (direction-independent weights, production kernels) was predicted from the model sky:

| Prediction | Residual rms |
|---|---|
| Direction-independent gains only | 210.7 Jy |
| Screens solved by kernel-StEFCal | 117.2 Jy |
| True screens | 112.3 Jy |

The remainder is sky-model error.

## 8.10 Conclusions and recommended methodology

1. **Real.** LWA-SV has per-stand direction-dependent phases of about 15–24° rms near 40 MHz, between the directions of Cyg A and Cas A. They are shared by the two dipoles of a stand, stable for at least 20 minutes and probably for hours, and are neither ionospheric nor geometric. Two independent datasets, 2.5 years apart, agree.
2. **They matter for fidelity, not detection.** Errors of this size leave static artefacts of about 10–15% of the single-channel thermal noise per image around each bright source. The diffuse Galaxy seen through the screens contributes errors comparable to that noise at 34.5 MHz. The artefacts are static, so a transient search that subtracts a background is not affected.
3. **Gridding kernels are the wrong place to correct them.** The production 5×5 kernels can represent screens smoother than about 15–20°, but a direct imager applies a kernel correction per image pixel, not per source direction. Only main lobes improve, and sidelobes get worse. Even perfect screens made the whole-sky error worse (151 → 307 Jy). Making this work would need the model-subtraction major cycles that EPIC, as a real-time dirty imager, does not have.
4. **They can be estimated.** Kernel-StEFCal, with the 25 kernel-space modes per stand solved against the whole forward-modelled sky and pooled over LST, recovers the screens to 1–2° in simulation.
5. **The gain comes from the image domain.** Predicting each image's static content with the solved screens, on the CPU or on a spare GPU, halves the residual against the actual image (211 → 117 Jy). This serves:
   - difference imaging for transients;
   - flux-scale and fidelity monitoring (section 4.6);
   - subtraction of the A-team, whose artefacts image-domain subtraction removes 3.5–6×.

**Recommended next steps.** None requires a change to the GPU imaging kernel.

1. Move the calibration correlation to the tensor-core correlator on the imaging GPU. **Done** (section 8.9.2, `--caltap_gpu`): every sample is visible to the solver, 20× the data of the 5% tap. Cross-polarisation products are computed but not yet published.
2. Run kernel-StEFCal on hours of those correlations to measure the real per-stand screens. Check their stability from night to night and between X and Y, and whether their structure is smooth enough to model.
3. Prototype the image-domain prediction and subtraction on the CPU. Its cost is one predicted image per channel per update, refreshed as the sky rotates, about once a minute. Measure the residual on real EPIC images.

Gridding-kernel corrections stay off the table unless EPIC gains a model-subtraction loop.

## 8.9 Tensor-core correlator

The ASTRON Tensor-Core Correlator (Romein 2021, https://git.astron.nl/RD/tensor-core-correlator) was built on epic1 at commit d772f81. It needs cudawrappers 1.0.0, because cudawrappers `main` requires CUDA ≥ 12.4 and the workers run 12.3. CMake points `FETCHCONTENT_SOURCE_DIR_CUDAWRAPPERS` at a 1.0.0 checkout.

| Test on the RTX 4090 | Result |
|---|---|
| `CorrelatorTest -I i4 -c 128 -n 256 -t 1024 -V 1` | 0 errors |
| One full gulp (128 channels × 1,024 samples × 512 inputs, all four polarisation products), 4-bit | 0.23 ms, 593 TOPS |
| The same, 8-bit | 0.31 ms, 441 TOPS |

`tests/streamcal/tcc/tcc_correlate.cc` is a file-driven correlator that accumulates on the GPU. `src/python/streamcal/tcc.py` wraps it; `tests/streamcal/test_tcc.py` checks it against numpy, exactly, for all four polarisation products. The TCC convention is $`v[b,p_Y,p_X]=\sum_t x_{Y,p_Y}\,\overline{x_{X,p_X}}`$ with $`b = X(X+1)/2+Y`$, $`Y\le X`$. TCC packs the real part in the low nibble and epic++ (`CNib`) the imaginary part, so raw F-engine bytes fed as i4 give the conjugate visibilities.

**Use in the calibration tap.** Correlating every sample of every gulp on the imaging GPU would cost about 0.23 ms for the correlation itself. A reorder from epic++'s `[chan][seq][stand][pol]` layout to TCC's `[chan][seq/32][stand][pol][32]` blocks, and on-GPU accumulation (`launchAsync(..., add = true)`), add a few tenths of a millisecond. That is well inside the 5 ms headroom of section 7.2, and it replaces the 5% CPU tap with the full data. Section 8.9.2 describes the integration in epic++.

### 8.9.1 Does the correlator get scheduled beside the imager?

The imaging kernel launches one 1,024-thread block per channel, 128 blocks on the 128 SMs, at the 64-register limit. While an imaging block is resident, its SM has no registers left for another block.

The test ran on epic1 (2026-10-17, RTX 4090 at default clocks) with two processes on the same GPU, time-sliced by the driver with no MPS (`research/bench_share.sh`):

- `epic++ --bench`: production imaging options, simulated 4+4-bit gulps, either paced at the F-engine rate or unpaced.
- `tests/streamcal/tcc/tcc_load`: the tensor-core correlator on a gulp-sized block (128 channels × 1,024 samples × 256 stands × 2 polarisations, 4-bit). It ran either once per gulp (41.8 ms) or back-to-back, and optionally copied the 65.5 MB block from pinned host memory before every correlation.

| Imager | Correlator | Imager per gulp, mean / p99 | Correlations completed | Correlator latency p50 / p99 / max |
|---|---|---|---|---|
| real time | none | 41.796 / 41.80 ms, no buffer waits | | |
| real time | one per gulp | 41.796 / 41.80–42.8 ms, no buffer waits | 958 of 958 | 1.6 / 2.7 / 3.2 ms |
| real time | back to back | 41.796 / 41.80 ms, no buffer waits | 401 per s (17× one per gulp) | 2.7 / 2.7 / 3.2 ms |
| real time | one per gulp, with copy | 41.796 / 41.80 ms, no buffer waits | 958 of 958 | 4.5 / 5.7 / 5.9 ms |
| unpaced | none | 35.89 / 36.8 ms | | |
| unpaced | one per gulp | 35.80 / 36.8 ms (36.4 in the first block) | 958 of 958 | 1.7 / 2.9 / 3.0 ms |
| unpaced | back to back | 35.82 / 36.7 ms (37.0 in the first block) | 401 per s | 2.7 / 2.7 / 3.4 ms |
| unpaced | one per gulp, with copy | 36.69 / 38.4 ms while correlating, 35.7 after | 958 of 958 | 4.6 / 7.1 / 7.5 ms |

**The correlator does get scheduled.** The imaging pipeline leaves the GPU partly idle within every gulp: host-to-device copies, the gaps between the 100-sequence chunk launches, and the CPU work between gulps. Time slicing fills those gaps.

- **Real time:** a full-rate correlation of every gulp, including copying the data, completed every time and never put the imager behind.
- **Unpaced (the worst case):** imaging slowed by about 0.8 ms per gulp (2%), and the p99 rose to 38.4 ms, still 3.4 ms inside the 41.8 ms of data.
- **Isolated spikes:** a few single gulps reached 46–48 ms. The 20 gulp buffers (0.8 s) absorbed them.

A production version as a separate process would read the voltages the imager already holds: either a pinned host copy of the gulp (the tap's staging), as in the "with copy" rows, or the imager's device buffer through CUDA IPC, which saves the copy. In-process integration would schedule differently (concurrent streams instead of time slicing) and would need its own test.

### 8.9.2 The tensor-core correlator as the calibration tap in epic++

`--caltap_gpu` replaces the 5% CPU slice with a correlation of every sample of every gulp on the imaging GPU (`src/ex/tcc_tap.h`, `tcc_tap.cu`). It is built when CMake is given `-DEPIC_TCC_ROOT` (a built TCC checkout) and `-DEPIC_CUDAWRAPPERS_INCLUDE`; otherwise the option stops epic++ with an error.

Per gulp, inside `ProcessGulp` and in the same process (concurrent streams, not time slicing):

1. The tap's own stream waits until every imaging stream has copied its channels to the device. The gulp is correlated from that copy, so the tap adds no host-to-device traffic.
2. A reorder kernel converts the 4+4-bit gulp `[chan][seq][stand][pol]` into the TCC 8-bit layout `[chan][seq/16][stand][pol][16][re, im]`, zero-padded from 1,000 to 1,008 sequences. The 8-bit format avoids the nibble-order difference of section 8.9.
3. TCC correlates and accumulates into an int32 buffer on the GPU. 4-bit samples give at most 128 per product, so integrations of up to 600 s cannot overflow.
4. Before the next gulp is copied over the device buffer, each imaging stream waits for the previous reorder, about 0.2 ms after its copy.
5. After `--caltap_interval` (10 s) the accumulator is copied to the host on a second stream. The next integration starts in the other buffer, and a publisher thread writes the XX and YY matrices to the same `EPICVIS1` table as the CPU tap (`src/ex/vis_table.hpp`, now shared by both taps). streamcal reads it unchanged; the header note is `tcc-i8 v1` and `duty` is 1.

The tap uses 383 MiB of GPU memory for 128 channels.

**Verification.** `tests/streamcal/test_tcc_tap.cu` (built by `tests/streamcal/tcc/build.sh`) feeds random gulps through the tap the way `ProcessGulp` does, reusing a single device buffer, and compares the table with the CPU X-engine: maximum difference 2.4 × 10⁻⁶ against |R| up to 44.5, which is float rounding. In the benchmark, the full-rate table equals numpy correlations of the stored gulps.

**Benchmark** (epic1, RTX 4090, production imaging options, simulated data regenerated with the current sky model as `model76`; `GPUTAP=1` in `bench_run.sh`):

| | CPU tap, 50 of 1,000 sequences | GPU tap, all sequences |
|---|---|---|
| Unpaced, per gulp | 35.8 ms | 36.2 ms |
| Real time, per gulp, mean / p99 | 41.796 / 41.80 ms, no buffer waits | 41.796 / 41.80 ms, no buffer waits |
| Samples per 10 s table and channel | 12,000 | 240,000 |
| Gain phase error, X / Y | 0.73° / 0.82° | 0.54° / 0.59° |
| Gain amplitude error, X / Y | 3.1% / 4.3% | 3.1% / 4.3% |
| A-team flux scales, Cyg A / Cas A / Tau A | 1.01 / 1.01 / 1.07 | 1.00 / 1.00 / 1.04 |
| χ²/dof, median | 1.01 | 3.02 |
| Image residual against exact gains | 0.024 | 0.024 |

The amplitude error is the same for both taps, so the Kalman tracker's bandpass model, not the data, sets it. The image residual is set by the same error.

**Sky-model errors set the floor at full rate.** The simulated sky contains 25 faint sources and 10% small-scale diffuse structure that the calibration model lacks. With the CPU tap these sit below the noise (χ² = 1.0). With 20× the samples they do not: χ² = 3.0. Part of that is an artefact, since the 96 stored gulps are replayed, so a 10 s table sums each sample about 2.5 times and the noise estimate is too low by that factor. Two consequences:

- The daemon's RFI rejection was absolute (`--max-chi2 3`) and rejected every channel at full rate. Channels and stands are now judged against the band and channel median χ², which leaves the CPU-tap behaviour unchanged.
- A first benchmark on the earlier dataset gave A-team fluxes of ×1.4–1.8 and 7–9% amplitude errors. That dataset was generated before the A-team was inpainted out of the LFSM map (section 8.5), so its true sky counted the A-team twice. At the CPU tap's noise level this was invisible. A full-rate tap will expose sky-model deficiencies in real data the same way, which matters for the direction-dependent screens (section 8.8). Bright compact sources beyond the A-team should be added to the model.
