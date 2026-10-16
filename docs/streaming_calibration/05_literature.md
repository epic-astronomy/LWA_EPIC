# 5. Literature: real-time calibration and direct-imaging correlators

This section summarises the published work that informs the design in sections 1 to 4. Identifiers were checked against arXiv or the publisher on 2026-10-16. Where no preprint could be located, this is stated.

## 5.1 Direct-imaging correlators and their calibration

| Work | Calibration approach | Relevance to EPIC |
|---|---|---|
| Morales 2011, MOFF ([arXiv:0812.3669](https://arxiv.org/abs/0812.3669)) | Calibrated voltages are assumed before gridding. A feedback loop from the image to the calibration block is sketched without an algorithm. | Origin of the EPICal idea. |
| Tegmark & Zaldarriaga 2009, FFT telescope ([arXiv:0805.4414](https://arxiv.org/abs/0805.4414)) | Relies on redundancy. | Not applicable to the pseudo-random LWA-SV layout. |
| Foster et al. 2014, BEST-2 ([arXiv:1401.6753](https://arxiv.org/abs/1401.6753)) | A parallel FX correlator derives the gains, which are applied before the spatial FFT. | The "visibilities on the side" pattern adopted here, made affordable by a duty-cycled CPU X-engine. |
| Zheng et al. 2014, MITEoR ([arXiv:1405.5527](https://arxiv.org/abs/1405.5527)) | Redundant calibration (omnical). | Needs redundancy. |
| Thyagarajan et al. 2017, EPIC ([arXiv:1510.08318](https://arxiv.org/abs/1510.08318)) | Gains must be applied before gridding (§3, "Calibration"). The package can apply pre-determined solutions or EPICal. | Formalism of section 1.1. |
| Beardsley et al. 2017, EPICal ([arXiv:1603.02126](https://arxiv.org/abs/1603.02126)) | Each antenna is correlated with the image pixel of a calibrator, the self term is subtracted, and gains are updated with damping γ. LWA1: 248 antennas, γ = 0.7, 51.2 ms loops, convergence in about 1.5 s; dynamic range improved from 80 to 133; only Cyg A and Cas A modelled. | Analysed in section 1.3. |
| Kent et al. 2019, EPIC on LWA-SV ([arXiv:1904.11422](https://arxiv.org/abs/1904.11422)) | Cable delays only; gain calibration left as future work. | The state that streamcal replaces. |
| Gorthi, Parsons & Dillon 2021 ([arXiv:2005.03060](https://arxiv.org/abs/2005.03060)) | Two O(N log N) schemes beside an FFT correlator: round-robin low-cadence correlation, and subset redundant calibration. Comments that EPICal needs accurate beam and diffuse-sky models. | The low-cadence idea transfers directly, with a sky-model solver in place of omnical. |
| Krishnan et al. 2023, MNRAS 520, 1928 | Commissioning of the commensal EPIC imager on LWA-SV. No calibration algorithm (no preprint located). | Context. |

## 5.2 Visibility-domain solvers

- **StEFCal** (Salvini & Wijnholds 2014, A&A 571, A97, [arXiv:1410.2101](https://arxiv.org/abs/1410.2101)).
  - Alternating per-antenna least squares, with the solution averaged on every second iteration.
  - Cost is about 20 P² flops per iteration, and the iteration count depends only weakly on P: about 12–20 iterations to a relative tolerance of 10⁻⁵.
  - In AARTFAAC, warm-starting from the previous snapshot gave an overall speed-up above 250×.
  - Smirnov & Tasse 2015 ([arXiv:1502.06974](https://arxiv.org/abs/1502.06974)) show StEFCal is a Gauss-Newton step with a diagonal approximation to the Hessian.
- **WALS multisource self-calibration** (Wijnholds & van der Veen 2009, [arXiv:1003.1892](https://arxiv.org/abs/1003.1892)).
  - Alternates between gains, receiver noise powers and apparent source powers.
  - It is the basis of LOFAR station calibration and AARTFAAC. The solution of free source fluxes (here, the Sun) follows the same idea.
- **SAGECal and consensus ADMM** (Yatawatta 2015, [arXiv:1502.00858](https://arxiv.org/abs/1502.00858)).
  - Per-frequency solutions are coupled by a smoothness penalty in frequency.
  - The delay parametrisation of the tracker in section 1.5 is a low-cost version of this.
- **Robust estimators**:
  - Student-t ECME (Kazemi & Yatawatta, [arXiv:1307.5040](https://arxiv.org/abs/1307.5040)).
  - Compound-Gaussian and relaxed concentrated maximum likelihood (Ollier et al., [arXiv:1603.01070](https://arxiv.org/abs/1603.01070), [arXiv:1612.02023](https://arxiv.org/abs/1612.02023)).
  - In this design, outliers are handled by the χ² gates of section 1.5.
- **Eigenvector (rank-1) calibration** (CHIME: Newburgh et al. 2014, [arXiv:1406.2267](https://arxiv.org/abs/1406.2267); CHIME/FRB, [arXiv:1901.04524](https://arxiv.org/abs/1901.04524)).
  - Gains are the dominant eigenvector during a point-source transit.
  - The all-sky LWA-SV covariance is far from rank 1 (diffuse Galactic emission plus several A-team sources), so this method serves only as an initial guess.

## 5.3 Recursive and tracking calibration

- **Tasse 2014**, non-linear Kalman filters for calibration (A&A 568, A67, [arXiv:1403.6308](https://arxiv.org/abs/1403.6308)).
  - The physical gain parameters are the state, the predicted visibilities the measurement model, and a random walk the process model.
  - The filter of section 1.5 is a diagonal, per-antenna reduction of this with phase and delay as the state.
- **KalCal** (B. Welman, Rhodes University MSc thesis 2024; [repository](https://researchrepository.ru.ac.za/handle/20.500.14915/3222)).
  - A Kalman filter with an RTS smoother and evidence-based tuning of the process noise.
  - It matches least-squares solution intervals at high S/N and outperforms them at low S/N.
- **AARTFAAC** (Prasad et al. 2014, A&A 568, A48; no preprint located).
  - Real-time model-sky calibration of up to 288 dipoles at about 60 MHz, with warm-started tracking.
  - Ionospheric phase errors were identified as the main impediment.
- **Kuiack et al. 2019** ([arXiv:1810.06430](https://arxiv.org/abs/1810.06430)): a flux scale for every 1 s image, bootstrapped against a 167-source catalogue. This is the model for the image-domain flux-scale monitor of section 4.6.
- **SKA SDP real-time calibration** ([documentation](https://skatelescope-ska-sdp-realtime-calibration.readthedocs.io/en/latest/)).
  - Gains are solved on the side and pushed to the beamformer through a message bus.
  - The architecture matches the CPU/GPU split adopted here.

## 5.4 O(N) antenna-to-beam correlation

- **Self-holography** (Wijnholds 2017, ICEAA; Wilke, Wijnholds & Gilmore 2022, JATIS 8, 011008): each path is correlated with a beam formed toward a calibrator.
- **Gueuning et al. 2020** ([arXiv:2005.11712](https://arxiv.org/abs/2005.11712)) show that, with the beam formed from the other antennas weighted by the current model, one pass equals one StEFCal iteration.
- **Kiefner et al. 2021** ([arXiv:2103.12492](https://arxiv.org/abs/2103.12492)): holographic calibration on EDA2.
- **Relation to EPICal.** EPICal is this method with the beam formed by the imager itself (section 1.3).

## 5.5 Stability of dipole-array gains and the ionosphere

- **EDA2** (Wayth et al. 2022, [arXiv:2112.00908](https://arxiv.org/abs/2112.00908)): delays are stable over months, and phases over hours to days.
- **Ionospheric phase structure.**
  - Mevius et al. 2016 ([arXiv:1606.04683](https://arxiv.org/abs/1606.04683)) give a phase structure function exponent β ≈ 1.89, with a diffractive scale of 3.5–30 km at 150 MHz.
  - Scaled to LWA-SV (about 100 m), this predicts about 0.7–5° rms differential phase at 60 MHz and up to about 10° at 30 MHz.
  - At this scale the ionosphere is nearly isoplanatic across the array but direction dependent across the sky: it appears mostly as small apparent source shifts rather than per-antenna gains.
- **Travelling ionospheric disturbances** (Helmboldt et al. 2012, [arXiv:1201.3872](https://arxiv.org/abs/1201.3872)): medium-scale TIDs with periods of 10–60 min dominate.
- **Measured on LWA-SV** (section 3.6): per-input changes of 5–7° in phase offset, 1.3–3 ns in delay and about 10% in amplitude over 2–4 hours.

## 5.6 Sky and beam models

- **Compact sources.**
  - Baars et al. 1977 (A&A 61, 99) is used in the prototype. Its secular-decrease law for Cas A overestimates the fading at low frequencies ([arXiv:1704.00002](https://arxiv.org/abs/1704.00002)).
  - Perley & Butler 2017 ([arXiv:1609.05940](https://arxiv.org/abs/1609.05940)).
  - **Recommended:** de Gasperin et al. 2020 ([arXiv:2002.10431](https://arxiv.org/abs/2002.10431)), LOFAR models of Cas A, Cyg A, Tau A and Vir A at 30–77 MHz.
- **Diffuse emission.**
  - LFSM (Dowell et al. 2017, [arXiv:1705.05819](https://arxiv.org/abs/1705.05819)), built from LWA1 maps at 35–80 MHz, is used through pygdsm.
  - GSM2016 (Zheng et al. 2017, [arXiv:1605.04920](https://arxiv.org/abs/1605.04920)) is the alternative.
- **Dipole beam:** LSL `lsl.sim.beam` (empirical and NEC models).
- **Simulators:** matvis ([docs](https://vis-cpu.readthedocs.io/en/latest/)) and fftvis (Cox et al. 2025) are faster alternatives to the GEMM forward model of section 2.4 if the diffuse model is made finer.

## 5.7 Other sources consulted

LOFAR station calibration cookbook (https://www.lofar-uk.org/files/statcal_cookbook.pdf); CHIME overview ([arXiv:2201.07869](https://arxiv.org/abs/2201.07869)); Jones & Wayth 2021 on non-identical element patterns ([arXiv:2105.06134](https://arxiv.org/abs/2105.06134)); SKA-Low station calibration with mutual coupling ([arXiv:2003.07306](https://arxiv.org/abs/2003.07306)); OVRO-LWA real-time pipeline (B. Byrne, LWA Users Meeting 2023, https://leo.phys.unm.edu/~lwa/users23/presentations/Byrne_lwa_meeting_presentation_ovro_lwa.pdf); EPIC validation against WSClean (C. Taylor, LWA Users Meeting 2023, https://leo.phys.unm.edu/~lwa/users23/presentations/Craig_Taylor-UsersMtg2023.pdf).
