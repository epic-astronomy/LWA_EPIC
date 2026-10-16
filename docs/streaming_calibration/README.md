# Streaming calibration for the EPIC imager

Notes and prototype for real-time per-antenna gain calibration of epic++ at LWA-SV. The gains are solved on the host CPUs and applied on the GPU. Work dated 2026-10-16, on branch `streaming-calibration` (base: `phoenix` v0.19.0, 670d6da).

## Summary

The EPIC imager (Thyagarajan et al. 2017) mixes all antennas before any averaging, so gains must be applied to the voltages before gridding. The GPU kernel already runs at about 95% of its 40 ms gulp budget and at the register limit. Calibration must therefore run on the CPU, and only its result may reach the GPU.

The adopted formalism (section 1.5) has five parts:

1. **Tap.** A tap inside epic++ correlates a small slice of every gulp on the CPU: 2.5–5% of the samples, with an AVX2 X-engine on the raw 4+4-bit voltages. Every 10 s it publishes the 256×256 XX and YY covariance of all 128 channels to shared memory. The tap reads the voltages before any gains are applied, so **there is no feedback loop**.
2. **Forward model.** A CPU service, `streamcal`, predicts the visibilities of the sky every 60 s: A-team, the LFSM diffuse sky at nside 64, and the Sun as a free flux, through the LSL dipole beam.
3. **Solution.** Each solved channel and polarisation is fitted with warm-started StEFCal. Channels whose χ²/dof shows RFI are rejected.
4. **Tracking.** A per-antenna Kalman filter tracks phase and delay; amplitudes and the bandpass are averaged slowly. Gains are interpolated to all channels.
5. **Application.** epic++ reads the gain table at image boundaries and folds it into `phases_g`, the per-antenna phases the GPU kernel already applies. Calibration costs the GPU nothing, and the autocorrelation removal stays consistent.

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

**Tests:** epic++ with the tap and gain table builds on epic1. The C++ shared-memory I/O tests (11/11), the Python–C++ round trip and the end-to-end daemon test pass (1.1° and 2.2% after 60 s). Nothing has been deployed; the imager was not started or restarted.

## Documents

| File | Content |
|---|---|
| [01_formalism.md](01_formalism.md) | Measurement model, exact EPIC image operator, EPICal as a projection, grid-domain solver, adopted streaming formalism, budgets |
| [02_forward_model.md](02_forward_model.md) | Sky components, spectra, beam, evaluation, omissions |
| [03_experiments.md](03_experiments.md) | All simulations, the LWA-SV data analysis, X-engine benchmark |
| [04_implementation.md](04_implementation.md) | epic++ changes, options, shared-memory layouts, streamcal modules, tests, commissioning plan, open items |
| [05_literature.md](05_literature.md) | Survey of real-time and direct-imaging calibration, with references |

## Code

| Path | Content |
|---|---|
| `src/ex/xengine.hpp`, `src/ex/cal_tap.hpp`, `src/ex/gain_table.hpp` | CPU X-engine, calibration tap, gain-table reader |
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
