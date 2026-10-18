# 10. Calibration methodology on real data: weights, source frame, refraction prior, interference

Sections 3.6, 8 and 9 measured the LWA-SV instrument on the EPIC Memo 12 TBF captures. This section records how the solution is turned into imaging weights and how the sky-model source positions are handled, as tested on the same captures imaged with the exact EPIC operator. It also records what the FRB science case requires of calibration.

All real-data results here come from the lower tuning of the Memo 12 captures (2023-09-17, 28–48 MHz). The 57–88 MHz bands that matter most for FRBs are covered only in simulation (section 10.6).

## 10.1 Data and imaging

- **Data:** the 32 captures were correlated in full on the GPUs of epic1–6 with the tensor-core correlator (`research/tbf_tcc.py`, `tbf_tcc_dispatch.sh`): about 25,000 samples per channel in each of the 792 channels, as even and odd halves.
- **Imaging:** 128×128 at 1.056° with `epicop.image_from_vis`, autocorrelations removed, as production.
- **Weights:** the per-stand weights are the nominal phases times either the cable model alone ("before") or the calibration corrections.
- **Sky model:** de Gasperin et al. 2020 A-team spectra and the OVRO-LWA hybrid diffuse map, the defaults of section 4.7.
- **Scripts and figures:** in the notes repository, `epic_notes/tbf_calibration_images/`.

## 10.2 Imaging weights: the matched filter, not 1/g

Applying $`c_a = 1/g_a`$ equalises the stands. A stand with excess receiver noise, or a fault, has a low gain, so 1/g up-weights its noise. On the TBF captures plain 1/g raised the thermal noise of the image by 16–25%.

The weight that maximises the signal-to-noise of a point source is the matched filter

```math
w_a \propto \frac{g_a^*}{R_{aa}} = \frac{1}{g_a}\,\rho_a,\qquad \rho_a = \frac{|g_a|^2 M_{aa}}{R_{aa}},
```

where $`\rho_a`$ is the fraction of stand $`a`$'s power that is sky (`solvers.sky_fraction`).
- For a sky-noise-dominated stand, $`\rho_a`$ is the same for all stands and $`w_a \propto 1/g_a`$.
- A stand with excess receiver noise has a lower $`\rho_a`$ and is down-weighted.

The daemon publishes $`c_a = (1/g_a)\min(1, \rho_a/\mathrm{median}\,\rho)`$, with $`\rho`$ the per-stand median over the solved channels, smoothed over tables. Then:
- healthy stands keep 1/g;
- the table normalisation (median |c| = 1) is unchanged;
- `--plain-weights` restores 1/g.

## 10.3 Source frame: anchoring and the refraction prior

**Anchoring.** A phase gradient across the stands shifts the whole image, and so does a common offset of all source positions. The two are degenerate in the solve. The daemon removes the flux-weighted mean source offset (`solvers.anchor_offsets`), so:
- the image frame stays on the catalogue positions;
- only differential refraction is left to the sources;
- any common shift goes into the gains.

`--free-frame` disables this.

**Refraction prior.** On the quiet-night captures, freely fitted source offsets of 0.17–0.51° (at 40 MHz) improved χ² by only 1–2%. Pinning the sources to the catalogue fitted equally well or better. The free offsets were absorbing direction-dependent phases and sky-model error, not refraction. During an ionospheric storm, however, real refraction of about 1° at 40 MHz must be followed (section 6).

`stefcal_sky(prior_off_deg, prior_neff, prior_nsamp)` adds a Gaussian prior on each offset, as Tikhonov rows in the linear sky step, with width $`\sigma_p`$ (daemon default 0.05° at 74 MHz, scaling as $`f^{-2}`$). Its weight separates the two parts of the misfit:

```math
\alpha^2 = \frac{1}{\sigma_p^2}\left(\frac{N_{\rm th}}{n_{\rm rows}} + \frac{\max(\chi^2_{\rm free} - N_{\rm th},\,0)}{N_{\rm eff}}\right),\qquad N_{\rm th} = \sum_{ab} \frac{R_{aa}R_{bb}}{T}.
```

- **Thermal noise** ($`N_{\rm th}`$, independent per visibility) enters with its true per-visibility weight.
- **The excess over the noise,** measured after an unregularised fit so that uncorrected refraction is not counted as model error, is treated as one systematic error with $`N_{\rm eff}`$ effective independent constraints. An offset $`d`$ is then accepted only if it removes more than $`(d/\sigma_p)^2/N_{\rm eff}`$ of the systematic misfit.
- Daemon defaults: $`N_{\rm eff}=100`$ (`--refraction-prior-neff`), and $`T`$ from the table.
- The anchored law is also pooled over tables with a 600 s time constant (`--refraction-tau`).

| Prior | Quiet TBF night, A-team annulus residual, 04:00 / 06:00 UTC | Storm simulation (77.6 MHz), phase error / image shift |
|---|---|---|
| none (free offsets) | 90 / 96 Jy | 0.75° / 0.020° |
| positions fixed | 79 / 80 Jy (pinned) | 0.87° / 0.120° |
| misfit all counted as systematic, $`N_{\rm eff}=100`$ | 77 / 74 Jy | 0.86° / 0.110° |
| **noise split, $`N_{\rm eff}=100`$ (default)** | **78 / 73 Jy** | **0.77° / 0.040°** |

The first two prior weightings tried failed in opposite directions:
- a per-visibility weight was swamped by the ~10⁵ visibilities per offset and had no effect;
- a weight from the whole misfit pinned the sources even in the noise-dominated storm.

On quiet nights, prior widths with $`N_{\rm eff}`$ from 30 to 300 all keep the benefit of pinning.

## 10.4 Before and after calibration

Band average 39–42 MHz, captures at 02:00, 04:00 and 06:00 UTC. "Old" is plain 1/g with free offsets; "new" is the matched filter, the anchored frame and the prior.

| Quantity | Before (cable model) | Old | New |
|---|---|---|---|
| A-team peak / model | 0.79–0.85 | 0.95–1.03 | 0.98–1.03 |
| Whole-sky residual rms | 238–270 Jy | 114–154 Jy | 88–103 Jy |
| A-team annulus residual rms | 175–257 Jy | 90–226 Jy | 73–110 Jy |
| Thermal noise (even − odd image) | 4.05–4.85 Jy | +16–25% | +3–8% |
| Peak S/N against thermal noise | | −5 to +13% | +12–18% |

- **Peak S/N:** the gain matches the 13–17% loss from cable-model errors estimated in section 3.6.
- **Total image rms** (the measure of Beardsley et al. 2017) **rises** after calibration, by 18–21%, to the model image's own rms. At 40 MHz the image variance is dominated by the Galactic diffuse emission, which calibration makes coherent just as it does the A-team; the artefact variance removed is about 3% of the total. A falling total rms is expected only where scattered compact flux dominates the image variance. The appropriate measures are the residual against the model and the S/N against thermal noise.
- **What is left:** compact residuals at Cyg A and Cas A, and a Cas A position excess of 0.14–0.23° beyond the peak-finder bias (0.05–0.22° for a perfect model). Both match the per-stand direction-dependent phases of section 8 and are independent of the prior and the frame. Section 10.7 lists the avenues under investigation.

## 10.5 What FRB detection requires of calibration

These requirements come from the study in `epic_notes/frb_sensitivity_and_calibration_requirements.md` and `frb_max_dm.md`. Those notes extrapolate the CHIME/FRB and LOFAR populations to the LWA-SV bands with $`\nu^{-4}`$ scattering, and give a rate of roughly 0.1–1.7 per year over 57–76 MHz at ideal noise. The rate is limited by sensitivity, not by calibration.

| Requirement | Target | Status |
|---|---|---|
| Per-stand phase and amplitude (≤ 5% rate loss) | ≲ 11°, ≲ 19%, no outlier stands | met by streaming calibration in simulation; the cable model in production loses 18–23% of the rate |
| Imaging weights | must not amplify noisy stands | met by the matched filter (section 10.2) |
| Image stability on 1–30 s | A-team scintillation and gain-table steps must stay below the dedispersed noise (0.15–0.7 Jy); A-team sidelobes are 43 Jy rms | open: A-team subtraction with gains toward each source (about 0.6°) and smooth interpolation between gain tables |
| Flux scale and beam (20% fluence) | each within 10% | open (section 9); not needed for detection |
| Position | ≤ 0.1–0.2° | met by calibration (image shift 0.01–0.04°); tilt and precessed WCS fixed in `phoenix` (23fcaeb) |

Two findings bear on the search itself rather than on calibration:
- Effective widths at 57–76 MHz are 7–21 s, longer than skymon's widest boxcar. Wider boxcars and a longer detrend window would raise the rate by 2–3×.
- Searching beyond DM ≈ 2000 adds less than about 1.5% of detectable events.

## 10.6 Validity in the upper band

Every real-data result above, and in sections 8.11, 9.7 and 4.7, comes from 28–48 MHz. For 57–88 MHz:

**Covered by simulation:**
- the imager benchmark (section 7.4: 0.54–0.59° with the GPU tap);
- the storm simulation (section 6 and 10.3).

Both are at 77.6 MHz.

**Expected to transfer:**
- refraction is about 3.5× smaller at 75 MHz than at 40 MHz;
- the A-team is relatively about 3× brighter against the Galaxy;
- the de Gasperin spectra reach 77 MHz and the OVRO-LWA maps 73 MHz.

**Must be measured in the upper band, not extrapolated:**
- the beam correction (fitted only to 45 MHz, and still growing with frequency);
- the direction-dependent screens;
- the prior's $`N_{\rm eff}`$ on quiet nights;
- the size of the matched-filter gain (the sky is about 10× colder at 75 MHz, so receiver noise matters more).

No archived high-band TBF/TBN data are known. The options are:
- GPU-tap tables recorded on epic4–6, which needs a production restart with `--caltap_gpu`;
- a new high-band capture;
- production FITS images.

## 10.7 Open: reducing the direction-dependent position excess

Under investigation, with train/test separation on the TBF captures:
1. the direction-dependent screens of section 8.11 in the model the gains are solved against;
2. per-direction gains toward each A-team source, for subtraction;
3. the beam correction of section 9.7 in the solve;
4. the sky model near Cas A (Galactic plane): the inpainting radius, the local diffuse gradient and nearby sources;
5. position metrics free of the peak-finder bias: a point-source fit in the visibilities, and PSF-matched correlation in the image.

## 10.8 Station-fixed interference at 75.8 MHz: subspace projection

**Data.** GPU-tap tables were recorded with `epic++ --vistap` on epic2 (75.8 MHz, 128 channels, 10 s tables of 240 000 spectra): 113 daytime tables from 2026-10-17 21:54 UTC and 29 night tables from 2026-10-18 02:10 UTC. Without mitigation, the daemon fits these tables to χ²/dof 510–570 by day and 360–440 at night, with 6–14 of 64 channel solutions rejected as RFI.

**The interferer.**
- It is broadband and always on, modulated at 120 Hz, and carries none of the cable delays: it couples into the signal path at the ARX boards, downstream of the antennas.
- In the whitened covariance its top eigenvalue is about 14 both by day and by night, and the day and night eigenvectors overlap to 0.994. Its spatial signature is fixed for hours.
- No sky direction explains it: the largest fraction of a component's power that a single sky direction captures (`rfi.sky_focus`) is 0.03–0.05, against values near 1 for compact sky sources.

**Method.** The interference occupies a low-rank subspace $`U`$ of the whitened covariance $`D^{-1/2} R D^{-1/2}`$, $`D = \mathrm{diag}(R)`$. The daemon solves the gains against the projected data with the oblique projector $`P = D^{1/2}(I - UU^H)D^{-1/2}`$, in a majorise–minimise loop that fills the projected-out part from the model:
$$R_{\rm fill} = R_{\rm mod} + P\,(R - R_{\rm mod})\,P^H,\qquad R_{\rm mod} = G M G^H\ \text{(diagonal: } \mathrm{diag}\,R - \mathrm{diag}\,\hat I).$$
`stefcal_sky` runs on $`R_{\rm fill}`$ with the source positions held; the fluxes stay free.

- **Cold start.** Without gains, a component is accepted only when its whitened eigenvalue exceeds both 4 and three times the largest eigenvalue of the whitened model (gain phases are a unitary similarity, so the sky's eigenvalues do not depend on them). On these tables this accepts one component.
- **Refresh.** On the next table, and then every `--rfi-refresh` tables (default 6), the subspace is re-estimated from the whitened residual $`D^{-1/2}(R - R_{\rm mod})D^{-1/2}`$ with the solved gains. Components above `--rfi-thresh` (default 0.5) and the sampling-noise edge, up to `--rfi-kmax` (default 4), are kept if their sky focus is below `--rfi-focus` (0.3). Flagged inputs are excluded from this residual: they carry model but no data. A refreshed subspace is accepted only if it lowers the projected χ²; otherwise the previous one is kept.
- Between refreshes the subspace is carried from table to table, and only `--rfi-mm` (default 4) MM iterations are run.

**Result** (18 consecutive tables, daemon replay with `research/replay_tables.py`):

| | Projection off | Projection on |
|---|---|---|
| Day, χ²/dof median | 508–569 | 35 (cold table), then 8.2–8.8 |
| Night, χ²/dof median | 360–441 | 40 (cold table), then 8.1–11.6 |
| Channel solutions rejected as RFI | 6–14 / 64 | 0–8 / 64 (median 0–1) |
| Components removed (median) | 0 | 4 |

The remaining χ²/dof of about 8 is consistent with the prototype study (5.9 with three components on a single table) and with the sky-model error floor at the full tap rate.

**Cost.** With 8 worker processes for 32 channels × 2 polarisations, a table solves in 2.0–2.5 s of wall time (about 1.6–2.5 s without projection), a refresh table in 5–10 s, and the cold table in 13–28 s. This is within the 10 s table interval on 8 cores, and leaves the remaining cores of a worker for the imager and the all-sky monitor.

**Projection of the voltages in the imager.** With `--proj PATH` the daemon publishes, after every gain table, a voltage-domain projector for all 128 imaged channels (section 4.3). The daemon's subspace $`U`$ lives in the whitened, cable-corrected basis of the used stands; on the raw voltages the projection is $`x \mapsto x - \tilde U(\tilde V^H x)`$ with $`\tilde U = D^{1/2}U/c`$ and $`\tilde V = \bar c\, U/D^{1/2}`$, $`c`$ the cable correction. Channels that the daemon does not solve take the whitened subspace of the nearest solved channel (at most two channels away) with their own powers $`D`$: on the 75.8 MHz tables the dominant component's overlap between channels two apart is 0.998, so 0.2% of its power leaks past the projector. epic++ `--proj_table` applies it on the GPU before imaging (`RfiStage`, section 4.2), with dithered re-quantisation to 4+4 bits, and optionally zeroes impulsive sequences (`--impulse_sigma`). The calibration tap keeps correlating the raw voltages, so the daemon continues to see, and track, the interference it removes.

Applied to the raw covariance of recorded tables, the published projector lowers the top whitened eigenvalue from 14.0 (X) and 12.6 (Y) to 2.3 and 2.2 after the cold-start table (one component) and to 1.1 and 0.8 from the second table on (four components), the level of the sky. Unsolved channels reach the same values as solved ones (1.10 and 1.10).

**Real time, imager and calibrator together** (epic2: RTX 4090, Threadripper PRO 5965WX; `research/bench_run.sh` with `PROJ=1 IMPULSE=6`). The benchmark voltages have the covariance of a recorded 75.8 MHz table (`research/make_table_gulps.py`: 240 gulps, one table of distinct samples, quantised to 4+4 bits at the recorded level), so they carry the real sky, gains and interferer. The daemon ran on the four physical cores of the calibration block and their siblings (CPUs 20–23, 44–47; 8 workers, one BLAS thread each).

| | Without the stage | Projection (q = 4) and impulse detection |
|---|---|---|
| GPU per gulp, imaging and GPU tap, unpaced | 36.4 ms | 37.3 ms (of 41.8 ms) |
| Paced at the F-engine rate, 10 min | | 12,000 gulps at 41.796 ms; the generator never waited for a buffer (mean 0.0004 ms); images every 83.6 ms (p99 87 ms) |
| Daemon per 10 s table | | 3.1–3.2 s; 8.4–8.5 s on subspace-refresh tables (every sixth); cold start 19.8 s |
| Tables processed | | every table after the cold start (56 of 56), each followed by a gain table and a projector |
| Impulse detector at 6σ | | 0% of sequences flagged on the Gaussian benchmark data |

The benchmark freezes the model time (`--fixed-time`), so the 10.5 s model refresh ran once. In operation it recurs every 60 s in the daemon's main process; a refresh that coincides with a subspace-refresh table occupies the daemon for about 19 s, which costs at most one table per minute (the tracker coasts over it). Averaged over a minute the daemon uses about 35 of the 60 s on the calibration block.

Images from the same data and calibrator, with and without the voltage projection (`research/bench_rfi_images.py`; noise from consecutive images, peak at the brightest source):

| Images | Noise | Brightest source, S/N |
|---|---|---|
| Without voltage projection | 8.05 × 10¹⁴ | 196 |
| With voltage projection (q = 4) | 5.00 × 10¹⁴ (−38%) | 261 (+33%) |

The source peak itself is 17% lower with the projection: the direction-dependent signal loss of a rank-4 null and the interference's own contribution to that pixel. The S/N, which sets detection, rises by a third.

The χ²/dof of the benchmark run (34 at the cold start, then 11.6 falling to 9.7, with 9–17 of 64 channel solutions rejected) is higher than on the recorded tables themselves (8.2–8.8, 0–1 rejected). Re-synthesised voltages count the table's sampling noise twice and are quantised a second time; the figure describes the benchmark data, not the method.

**Solar bursts and the impulse detector.** The detector tests band power only, so it cannot tell a local impulse from a solar burst. Slow solar emission raises the whole gulp and is not flagged, but a type III burst near 75 MHz drifts at roughly 28 MHz/s (the empirical law of Alvarez & Haddock 1973) and crosses a 16-channel (0.38 MHz) sub-band in about 14 ms: a short spike that the detector would zero. The daemon therefore also publishes the Sun's response on the raw inputs, after projection and normalised per polarisation: $`w = Pv/\lVert Pv\rVert`$ with $`v = g\,a(\hat s_\odot)/c`$ from the tracked gains (only once they exist; with the cable model alone the beam does not focus on the Sun). The GPU kernel forms $`|w^H x|^2`$ for every channel and sequence, in the same block reduction as the band power, and the detector tests the band power minus this Sun-beam power. A solar burst raises both by the same amount and is kept; an impulse from elsewhere raises only the band power. The Sun-beam power also carries one input's worth of noise (about 0.4% of the band power), a constant offset for the test.

- Unit test (`test_rfi_stage.py`): a four-sequence burst along a Sun-like response, strong enough to be zeroed with the impulses when the table has no Sun vector, is kept when it has one, while the impulses are still zeroed.
- Recorded 75.8 MHz day table (Sun at 27.5° elevation): the cross-power of the Sun beam is 6.2 (X) and 8.3 (Y) times the median beam toward other directions at the same elevation (Cyg A: 7.0 and 7.9); with the cable model alone 0.9 and 1.2.
- Cost: GPU 37.6 ms per 41.8 ms gulp with projection, impulse detection and the Sun beam (37.3 ms without the Sun beam).

The exemption is as good as the Sun vector. Gain errors of 1–2° and 3%, the Sun's position shift by refraction (0.05–0.3° against a 2–3° synthesised beam) and the size of the burst source leave an estimated 1–2% of a burst's power outside the beam, so only bursts some 50–100 times above the detection threshold can still be partly zeroed. Below the horizon there is no Sun vector and the detector works on band power alone.

**Limits.**
- 38–49 inputs per polarisation are still power-flagged, because the interference inflates the autocorrelations (section 4.7).
- The subspace is assumed fixed between refreshes. A transmitter that moves or switches faster than `--rfi-refresh` tables is caught only by the χ² channel rejection.
- The interferer itself should be located and removed at the station.
