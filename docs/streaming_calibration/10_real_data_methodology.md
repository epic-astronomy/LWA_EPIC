# 10. Calibration methodology on real data: weights, source frame, refraction prior

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
