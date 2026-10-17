# Streaming calibration for the EPIC imager

Notes and prototype for real-time per-antenna gain calibration of epic++ at LWA-SV. The gains are solved on the host CPUs and applied on the GPU. Work dated 2026-10-16, on branch `streaming-calibration` (base: `phoenix` v0.19.0, 670d6da).

## Summary

The EPIC imager (Thyagarajan et al. 2017) mixes all antennas before any averaging, so gains must be applied to the voltages before gridding. The GPU kernel already runs at about 95% of its 40 ms gulp budget and at the register limit. Calibration must therefore run on the CPU, and only its result may reach the GPU.

The adopted formalism (section 1.5) has five parts:

1. **Tap.** A tap inside epic++ correlates the gulps and publishes the 256×256 XX and YY covariance of all 128 channels to shared memory every 10 s. The tap reads the voltages before any gains are applied, so **there is no feedback loop**. Two forms:
   - on the CPU, a slice of every gulp (2.5–5% of the samples) with an AVX2 X-engine on the raw 4+4-bit voltages;
   - on the GPU, every sample, with the tensor-core correlator (`--caltap_gpu`, section 8.9.2).
2. **Forward model.** A CPU service, `streamcal`, predicts the visibilities of the sky every 60 s through the LSL dipole beam:
   - the A-team, with de Gasperin et al. 2020 spectra;
   - the diffuse sky: the OVRO-LWA hybrid map at 30–80 MHz, LFSM outside;
   - the Sun as a free flux.

   An optional beam correction fitted to real data is described in section 9.7.
3. **Solution.** Each solved channel and polarisation is fitted with warm-started StEFCal. The flux and position of every bright source (A-team, Sun) are solved together with the gains, so scintillation, solar bursts and ionospheric refraction stay in the sky model rather than in the gains (section 6).
   - The source frame is anchored to the catalogue positions, and the offsets carry a refraction prior that separates thermal noise from model error (section 10.3).
   - Channels whose χ²/dof, relative to the band median, shows RFI are rejected.
4. **Tracking.** A per-antenna Kalman filter tracks phase and delay; amplitudes and the bandpass are averaged slowly. Gains are interpolated to all channels.
5. **Application.** epic++ reads the gain table at image boundaries and folds it into `phases_g`, the per-antenna phases the GPU kernel already applies. Calibration costs the GPU nothing, and the autocorrelation removal stays consistent. The published weights are the matched filter $`(1/g)\min(1,\rho/\bar\rho)`$, not plain 1/g, so stands with excess receiver noise are down-weighted (section 10.2).

## Main results

**EPICal** (Beardsley et al. 2017) is a projection of the covariance onto one beam formed by the imager (section 1.3). In epic++ it would need complex per-sequence pixels from inside the saturated kernel. In simulation, with that pixel supplied exactly, it reached 2.95° at night (amplitude error 27%) and 9.4° in daytime (section 3.4).

**Calibration from the EPIC images alone** (inverting the exact image operator, section 1.4) is well posed but weak:

- the images carry the gain information of about a 10% voltage tap;
- they are 5–10× more sensitive to sky-model error than the covariance;
- they fail at 34.5 MHz (11.5° at night, 26–79° in daytime) where StEFCal on the covariance gives 0.25–0.41° (section 3.2).

**Closed-loop simulation** (16 channels, 229 stands, production imaging geometry; section 3.4):

| Case | Tap route | Images route | EPICal | Cable model only |
|---|---|---|---|---|
| Night, 75.8 MHz | 2.5° (image residual 0.020) | 5.8° | 3.0° | 45.6° |
| Day, 34.5 MHz, Sun 3× model | 2.0–2.6° (Sun scale 2.97–3.04 recovered) | diverges | 9.4° | 44.5° |

The tap route used a 2% tap. It recovers from 120° jumps in 20–30 s and flags dead stands automatically.

**LWA-SV data** (Memo 12 TBF captures, 28–47 MHz; section 3.6):

- The forward model and StEFCal fit the measured covariance to χ²/dof 1.1–1.9 in clean channels, near the thermal-noise limit.
- RFI channels (χ²/dof 6–4,657) are identified by the same statistic.
- The cable-model-only weights epic++ uses are off by 6–8° in phase (MAD), 1.1 ns in delay and 10–12% in amplitude, with a heavy tail and 0.5–2% unflagged dead inputs. That costs 13–17% of point-source S/N.
- Gains drift on hour time scales.

**CPU cost** (sections 1.6 and 3.5):

| Item | Cost |
|---|---|
| X-engine | 13.1 GCMAC/s per Zen 3 core; a 5% tap needs 0.77 of a core |
| Solve | 0.4 s per 10 s table |
| Model refresh | 12 s per minute (32 channels × 2 polarisations, 8 Xeon threads) |

**GPU cost:** none in the kernel; a 512 kB copy between gulps when the table changes.

**Ionosphere and the Sun** (section 6). With the bright-source fluxes fixed, a 30× solar burst destroys the gain solution (90°). With fluxes and positions solved, the same burst costs 0.02°. In a closed-loop storm (scintillation σ = 0.3, refraction up to 1.4°, a 30× burst, 30% absorption), the applied gains held:

| Band | Phase error | Image shift |
|---|---|---|
| 34.5 MHz | 0.56° median, 0.79° worst | 0.016° |
| 75.8 MHz | 0.65° | 0.006° |

Without position solving, the shift was 0.24° and the burst pushed the error to 2.7°. The gains are direction-independent; the direction-dependent effects are parameters of the sky model (section 6.4).

**Imager and calibrator benchmark** (section 7). epic++ was run in its benchmark mode on epic1 with production imaging options and simulated 4+4-bit F-engine data with wrong gains (±70°, 3 ns, 20%, three dead inputs). The tap, the daemon and the gain table ran in the loop.

| Configuration | Per gulp, mean / p99 |
|---|---|
| Baseline (empty gulps, no calibration) | 35.5 / 35.9 ms |
| Full calibration loop | 35.8 / 36.4–36.8 ms |

- **Real time:** a 41.8 ms gulp leaves 5 ms of headroom, and in a 5-minute run paced at the real-time rate the imager never fell behind.
- **Gains:** the first table (10 s) brought them to 0.7°/5%, and they held at 0.6°/3–4%.
- **Images:** the residual of the GPU images against images made with the exact gains fell from 0.219 to 0.019.

The benchmark also found and fixed six defects, among them a buffer-release race in `buffer.hpp`, which the production pipeline is exposed to too, and an amplitude-scale bug that would have overflowed the GPU's half-precision FFT.

**Direction-dependent gains** (section 8):

- **Real:** LWA-SV stands show per-stand direction-dependent phases of 15–24° rms between Cyg A and Cas A near 40 MHz. They are stable for at least 20 minutes, shared by the two dipoles of a stand, and neither ionospheric nor geometric; two datasets 2.5 years apart agree.
- **Not correctable through kernels:** per-antenna gridding kernels can represent them but cannot apply them in a dirty-image correlator. A kernel acts per image pixel, not per source direction, so only main lobes improve; even perfect screens made the whole-sky error worse (151 → 307 Jy).
- **Useful in the image domain:** kernel-StEFCal estimates the whole-sky screens to 1–2° in simulation. Using them to predict the static image halves the residual (211 → 117 Jy), which is the route for difference imaging and A-team subtraction.
- **Tensor-core correlator:** built on the workers. It correlates a full gulp in 0.23 ms (4-bit). Run beside the imager on the same GPU, it correlated every gulp, including the 65.5 MB host-to-device copy, with the imager staying exactly real time; in the unpaced worst case imaging slowed by 2% (p99 38.4 ms of the 41.8 ms budget). The calibration tap can therefore move to 100% of the data on the GPU.

**GPU calibration tap** (section 8.9.2):
- `--caltap_gpu` correlates every sample on the imaging GPU in about 0.4 ms per gulp, and the imager stays real time.
- It matches the CPU X-engine to float rounding.
- In the benchmark the gains reach 0.54–0.59°, against 0.73–0.82° with the 5% CPU tap.
- At full rate the sky model, not noise, sets the error floor (χ²/dof about 3).

**Real-data methodology** (sections 4.7, 8.11, 9.7 and 10, on the 2023 TBF captures at 28–48 MHz):
- **Sky model:** the de Gasperin A-team spectra make the fitted Cas A / Cyg A scale ratio 0.96 (Baars: 1.70). The OVRO-LWA diffuse map and an azimuthal beam correction each lower the excess χ² by about 20%.
- **Direction-dependent screens:** kernel-StEFCal finds a station-fixed component that improves held-out data on a rotated sky by about 7% (8 smooth modes).
- **Imaging the captures after calibration:**
  - residual against the model 2.6–2.7× lower than with the cable model;
  - peak S/N +12–18%;
  - thermal noise +3–8% with the matched-filter weights (plain 1/g: +16–25%).
- **Refraction prior:** pins the sources on quiet nights and follows a simulated storm (0.77° and 0.04°, against 0.75° and 0.02° free).
- **Upper bands:** all real-data results are at 28–48 MHz. The 57–88 MHz bands need their own data (section 10.6).

**FRB requirements** (section 10.5): the expected rate (about 0.1–1.7 per year at 57–76 MHz) is limited by sensitivity, not calibration. The calibration requirements that matter for detection are:
- no outlier stands and noise-preserving weights, both met;
- image stability over 1–30 s, which needs A-team subtraction and smooth gain interpolation, still open.

**Primary beam** (section 9): epic++ applies no primary-beam correction. The dipole beam changes the apparent brightness of the watched sources by ×3–8 over a day and differs between X and Y by up to 1.7×. Correction is not needed for detection, which works in S/N, but is needed for fluxes, fluences, light curves and spectra. It belongs downstream of the GPU (pixel extractor, FITS), with the beam model refined from the A-team flux scales the calibration already solves.
- **Real-data fixes to the forward model:** the A-team removed from the LFSM map (it was counted twice); guard rails on the free source fluxes and positions.

**Tests:** epic++ with the tap, the gain table and the benchmark mode builds on epic1. The C++ shared-memory I/O tests (13/13), the Python–C++ round trip, the end-to-end daemon test (1.1° and 2.2% after 60 s) and the imager integration test (section 7) pass. The production imager was not deployed, started or restarted; the benchmark ran the branch build on an idle GPU.

## Documents

| File | Content |
|---|---|
| [01_formalism.md](01_formalism.md) | Measurement model, exact EPIC image operator, EPICal as a projection, grid-domain solver, adopted streaming formalism, budgets |
| [02_forward_model.md](02_forward_model.md) | Sky components, spectra, beam, evaluation, omissions |
| [03_experiments.md](03_experiments.md) | All simulations, the LWA-SV data analysis, X-engine benchmark |
| [04_implementation.md](04_implementation.md) | epic++ changes, options, shared-memory layouts, streamcal modules, tests, commissioning plan, open items |
| [05_literature.md](05_literature.md) | Survey of real-time and direct-imaging calibration, with references |
| [06_ionosphere.md](06_ionosphere.md) | Scintillation, refraction, absorption and solar bursts; sky parameters solved with the gains; storm simulations; direction dependence |
| [07_benchmark.md](07_benchmark.md) | epic++ benchmark mode, timing with and without calibration, end-to-end calibration of simulated F-engine data through the imager, defects found |
| [08_direction_dependent.md](08_direction_dependent.md) | Per-antenna direction-dependent gains: kernel formalism, representability, cost of uncorrected errors, measurement method, LWA-SV measurements (TBF 2023, TBN 2021), why kernels cannot apply them in a dirty imager, kernel-StEFCal and the real-data pilot, image-domain prediction, tensor-core correlator and the GPU calibration tap (`--caltap_gpu`) |
| [09_primary_beam.md](09_primary_beam.md) | Primary-beam correction: what EPIC images contain, beam variation for the watched sources (×3–8 over a day), when it is needed (flux, fluence, light curves; not detection), how to apply it, beam-model accuracy, the beam correction fitted to the TBF captures |
| [10_real_data_methodology.md](10_real_data_methodology.md) | Matched-filter imaging weights, the catalogue-anchored source frame and the refraction prior; before/after images of the TBF captures; FRB-driven calibration requirements; validity in the upper band; open work on the direction-dependent position excess |

## Code

| Path | Content |
|---|---|
| `src/ex/xengine.hpp`, `src/ex/cal_tap.hpp`, `src/ex/gain_table.hpp` | CPU X-engine, calibration tap, gain-table reader |
| `src/raft_kernels/bench_kernels.hpp`, `src/raft_kernels/epic_executor.hpp` | `epic++ --bench` (generator → correlator → sink) |
| `src/ex/buffer.hpp` | Fix: reset a buffer before releasing it |
| `src/raft_kernels/correlator.hpp`, `src/ex/MOFF_correlator.hpp`, `src/MOFF_cu_handler.cu` | Hooks: tap offer, gain polling at image boundaries, in-place phase upload, `gain_version` metadata |
| `src/python/streamcal/` | `daemon.py` (service), `forward.py`, `sky.py`, `solvers.py`, `epicop.py`, `shm.py`, `geometry.py` |
| `src/python/streamcal/research/` | Simulation and real-data scripts behind section 3 |
| `tests/streamcal/` | `test_io.cpp`, `test_shm_roundtrip.py`, `test_daemon.py` |

## Reproduction

Environment: the epic310 conda environment, or any Python 3.10 with numpy, scipy, astropy, healpy, `lsl` 3.x, `pygdsm` and `numba`.

```bash
# C++ I/O tests (no CUDA)
g++ -O3 -march=znver3 -fopenmp -std=c++17 -I tests/streamcal/shim -I src/ex \
    tests/streamcal/test_io.cpp -o /tmp/streamcal_test_io -lpthread
/tmp/streamcal_test_io selftest /dev/shm
python tests/streamcal/test_shm_roundtrip.py /tmp/streamcal_test_io /dev/shm/streamcal_test

# service, end to end on synthetic tap tables
python tests/streamcal/test_daemon.py /dev/shm/streamcal_e2e

# closed-loop simulation (section 3.4)
cd src/python/streamcal/research
python sim_loop.py --out sim_75.npz                                    # 75.8 MHz, night
python sim_loop.py --chan0 1376 --duration 600 --start 2026-10-16T19:00:00 \
       --sun_factor 3 --free_sun --out sim_34day.npz                   # 34.5 MHz, day
python plot_sim.py sim_75.npz sim_75.png

# LWA-SV TBF data (section 3.6)
python tbf_cache.py /FileStore/epic0/memo12_Data/rawfiles/060204_008031743_DeMux ../tbfR_1.npz 10000
python tbf_stability.py
```

Running the service beside an imager:

```bash
epic++ ... --caltap_nseq 25 --caltap_interval 10 [--gain_table /dev/shm/epic_gains_gpu{gpu}]
PYTHONPATH=src/python python -m streamcal.daemon --vis /dev/shm/epic_vis_gpu0 \
       --gains /dev/shm/epic_gains_gpu0 --dry-run
```

The commissioning order (tap only, then dry run, then the NDP sign check, then apply) is in section 4.6.
