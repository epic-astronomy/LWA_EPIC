# 4. Implementation: CPU solution, GPU application

All code is on branch `streaming-calibration` (from `phoenix` at v0.19.0, 670d6da). New features are off by default: an epic++ started without the new options behaves exactly as v0.19.0.

## 4.1 Data flow

```
F-engine packets ──> packet assembler ──> gulp [chan 128][seq 1000][stand 256][pol 2] (4+4 bit)
                                              │
              ┌───────────────────────────────┼──────────────────────────────────┐
              │ CorrelatorRft::run()          │                                  │
              │  image start: GainTableReader::Poll ──> ApplyGainTable ──> phases_g (GPU)
              │  every gulp:  CalTap::Offer(gulp)  (non-blocking; skipped if busy)
              │               ProcessGulp (GPU, ~38 ms)   ║  CalTap worker: stage 50 seq/chan (~3 MB)
              │               CalTap::WaitCopied()        ║                X-engine (AVX2, OpenMP)
              └───────────────────────────────────────────╨──────────────────────────────────┘
                                                                  │ every --caltap_interval s
                                       /dev/shm/epic_vis_gpu<N>   ▼  (seqlock, 134 MB)
                                                         streamcal daemon (Python, CPU)
                                                           forward model (60 s), StEFCal,
                                                           χ² gates, Kalman tracker
                                       /dev/shm/epic_gains_gpu<N> ▼  (seqlock, 256 kB)
                                              epic++ GainTableReader (next image boundary)
```

## 4.2 epic++ changes

| File | Change |
|---|---|
| `src/ex/gain_table.hpp` (new) | `GainTableHeader` (128 bytes) and `GainTableReader`, which reads under a seqlock. A table is rejected if the magic or layout is wrong, if the version is not newer, if `valid_from_seq` lies in the future, if it does not cover the imaged channels, or if it contains non-finite values or values above `max_abs` (64). Also `ApplyGainCorrections`, which multiplies nominal × table. |
| `src/ex/xengine.hpp` (new) | CPU X-engine: 4+4-bit unpack to int16 pairs, `vpmaddwd` in 2×3 register tiles, OpenMP over channels. Output is int64 accumulators for the XX and YY upper triangles. |
| `src/ex/cal_tap.hpp` (new) | `CalTap` has a one-slot mailbox, a stager thread and an X-engine thread. The stager copies `nseq` sequences per channel from the offered gulp into one of two staging batches while `ProcessGulp` runs; the correlator calls `WaitCopied()` before releasing the gulp. A full batch (`--caltap_batch` gulps, default 8) goes to the X-engine thread. Correlating one gulp per call kept up with only every second gulp, because the read-modify-write of the 134 MB accumulator dominates at 50 samples; batching keeps up with every gulp (section 7). The X-engine thread publishes `VisTableHeader` plus R as complex64 (Hermitian, R = Σxxᴴ / nsamp) under a seqlock. No `Payload` is shared across threads, because `Payload` unlocks its buffer on `use_count() == 2`, which is not thread safe. |
| `src/ex/MOFF_cu_handler.{h,cu}` | `UpdatePhases`: an in-place synchronous `cudaMemcpy` into the existing `m_phases_cu`, with no reallocation. It is safe between gulps because `ProcessGulp` returns only after all streams finish. |
| `src/ex/MOFF_correlator.hpp` | Keeps `m_phases_nominal` (from `GetLwasvPhases`). `ApplyGainTable` sets `m_phases = nominal ⊙ table` and uploads it. A new spectral window resets the gain version to 0, i.e. nominal phases. Adds `GetGainVersion` and `GetCalDesc`. |
| `src/raft_kernels/correlator.hpp` | Polls the gain table at the first gulp of every image, so each image is made with exactly one table. Offers every gulp to the tap. Adds `gain_version` (uint64) to the image metadata. |
| `src/ex/types.hpp`, `option_parser.hpp`, `kernel_types.hpp` | New options (below). `{gpu}` in paths is replaced by the device id. |
| `CMakeLists.txt` | Links `OpenMP::OpenMP_CXX` into epic++, for the X-engine. |
| `src/ex/buffer.hpp` (fix) | `Payload::unlock` released the buffer (`m_lock = 0`) before `ResetBuf()` cleared its metadata. A producer acquiring the buffer in that window wrote metadata concurrently with the reset and crashed in `unordered_map::operator[]`. It reproduced reliably with the benchmark generator under backpressure; the production packet assembler is exposed to the same race. Fixed by resetting first, then releasing with release ordering. |
| `src/raft_kernels/bench_kernels.hpp` (new), `epic_executor.hpp` | `epic++ --bench`: `BenchGulpGen` → `CorrelatorRft` (with tap and gain table) → `BenchSink`. Only these kernels are constructed, so nothing connects to the database, video server or network. The generator plays simulated gulps from a file, rotating each channel's sequences on every pass so the tap sees new samples, or sends empty gulps; it runs unpaced or at the real-time rate. The sink reports image cadence and gain version and dumps sampled images. |

New options:

| Option | Default | Meaning |
|---|---|---|
| `--gain_table PATH` | empty (off) | Gain table to apply, e.g. `/dev/shm/epic_gains_gpu{gpu}` |
| `--caltap_nseq N` | 0 (off) | Sequences of every 1000-sequence gulp that are correlated (50 = 5%) |
| `--caltap_out PATH` | `/dev/shm/epic_vis_gpu{gpu}` | Visibility table output |
| `--caltap_interval S` | 10 | Integration per published table, in seconds of F-engine time |
| `--caltap_threads N` | 2 | X-engine threads (not pinned; the raft kernels use CPUs 3–11 for GPU 0) |
| `--caltap_batch N` | 8 | Gulps staged per X-engine call |
| `--bench`, `--bench_data`, `--bench_seq0`, `--bench_ngulps`, `--bench_realtime`, `--bench_rotate`, `--bench_dump`, `--bench_dump_every`, `--bench_report` | off | Benchmark mode (section 7) |

## 4.3 Shared-memory layouts

Both tables are little-endian files in `/dev/shm` with a 128-byte header. The `seq` field is odd while the writer is updating the table. A reader copies the header and data, then re-reads `seq` and discards the copy if it changed. The Python twins are in `src/python/streamcal/shm.py`.

**Gain table** (`EPICGAIN`, layout 1). The corrections must be relative: the reader rejects a table whose median non-zero |c| lies outside 0.25–4, and the Python writer refuses to write one. The solved gains carry an absolute scale (F-engine counts per Jy^½, about 0.01). Publishing 1/g directly would scale the voltages by about 100 and overflow the half-precision gridding and FFT. In the integration test a table clipped at |c| = 32 turned every image into NaN (section 7.3). The daemon normalises each channel and polarisation to median |c| = 1 and keeps the absolute scale for the flux-scale monitor.

| Offset | Field |
|---|---|
| 0 | `magic[8]` |
| 8 | `layout`, `header_bytes` (u32 each) |
| 16 | `seq` (u64) |
| 24 | `version` (u64) |
| 32 | `chan0` (i64) |
| 40 | `nchan`, `nstand`, `npol`, `flags` (u32 each; flags bit 0 = valid, 0 = revert to nominal) |
| 56 | `valid_from_seq` (u64) |
| 64 | `t_solved_unix` (f64) |
| 72 | `max_abs` (f32), reserved (u32) |
| 80 | `solver[48]` |
| 128 | data: complex64 `[nchan][256][2]` |

**Visibility table** (`EPICVIS1`, layout 1):

| Offset | Field |
|---|---|
| 0 | `magic[8]` |
| 8 | `layout`, `header_bytes` (u32 each) |
| 16 | `seq` (u64) |
| 24 | `version` (u64) |
| 32 | `chan0` (i64) |
| 40 | `nchan`, `nstand`, `npol`, `flags` (u32 each) |
| 56 | `nsamp` (u64) |
| 64 | `seq_first` (u64) |
| 72 | `seq_last` (u64) |
| 80 | `ngulps`, `nskipped` (u32 each) |
| 88 | `gulp_nseq` (u64) |
| 96 | `duty` (f32) |
| 100 | `note[28]` |
| 128 | data: complex64 `[nchan][2][256][256]` |

The visibility table holds raw F-engine correlations, before cable model and phases. It occupies 134 MB per instance.

## 4.4 streamcal (Python)

| Module | Content |
|---|---|
| `geometry.py` | Wraps `epic_geometry` (stands, plane frame, used-stand mask), `CHAN_BW` |
| `sky.py` | A-team spectra (Baars et al. 1977 with Cas A decay; Tau A, Vir A low-frequency fits), quiet Sun, LFSM diffuse sky through pygdsm, optional small-scale fluctuations for simulation |
| `forward.py` | Sky snapshot (directions and beam-weighted fluxes per component) and model matrices $`M = A\,\mathrm{diag}(BS)\,A^H`$ as complex64 GEMMs |
| `solvers.py` | `stefcal` (with free fluxes), `gridcal` (grid-domain Gauss-Newton/CG), `Tracker` (Kalman filter on phase and delay, amplitude and bandpass smoothing, jump detection) |
| `epicop.py` | Exact epic++ image operator (scatter and gather, numba) for the grid-domain solver and the image monitor |
| `shm.py` | `GainTableWriter`, `VisTableReader` |
| `daemon.py` | `python -m streamcal.daemon --vis … --gains … [--dry-run]`. Two passes per table. The first solves gains, fluxes and positions per channel; on a cold start, gains are solved first against the nominal sky. The source offsets are then fitted to $`(f_c/f)^2`$ across the band. The second pass re-solves gains and fluxes with positions fixed. Channel jobs run in a process pool (`--workers`, default 8). The Kalman time step comes from the data timestamps. |
| `research/` | Every simulation and real-data script behind sections 3, 6 and 7, including the benchmark tools (`make_bench_model.py`, `make_bench_gulps.py`, `bench_run.sh`, `bench_watch.py`, `bench_images.py`) |

Dependencies beyond the epic310 environment: `pygdsm`, plus `numba` for `epicop` only. `lsl` provides the dipole beam and the cable model. The daemon imports neither numba nor the GPU stack.

## 4.5 Tests

| Test | Result (2026-10-16) |
|---|---|
| `tests/streamcal/test_io.cpp selftest` (epic1, g++ 11.4) | 13/13 pass: the tap R (batched, two threads) matches a scalar reference; the gain reader handles a missing file, slicing, repeated versions, a torn seqlock, deferral by `valid_from_seq`, coverage, NaN, clamping and unnormalised tables; application and zero flags are correct |
| `tests/streamcal/test_shm_roundtrip.py` (epic1, epic310 Python) | Pass: Python writer → C++ reader, and C++ tap → Python reader |
| `tests/streamcal/test_daemon.py` | End to end on synthetic tap tables: see section 3.5 |
| `research/test_epicop.py` | epicop against literal MOFF: relative error 2.5×10⁻⁸; adjoint exact |
| Full epic++ build (epic1, CUDA 12.3, `build-streamcal/`) | Builds; no new warnings |
| `epic++ --bench` with simulated F-engine data, tap, daemon and gain table (section 7) | Real time kept (35.8 ms per 41.8 ms gulp); gains to 0.6° / 3–4% from ±70° / 20%; GPU image residual against exact gains 0.219 → 0.019 |

The X-engine benchmark is `research`-level code in `tests/streamcal/test_io.cpp` (tap mode); throughput was measured with the standalone bench (section 1.6).

## 4.6 Commissioning plan

None of these steps has been carried out. Each starts or reconfigures the imager and must be scheduled with the operators.

1. **Tap only.**
   - Deploy the branch build on one node with `--caltap_nseq 25 --caltap_interval 10`.
   - Check that `BlockEpicImager` exec_time and the gulp rate are unchanged (the tap runs on its own threads), and watch `nskipped`.
2. **Dry run.**
   - Run `streamcal.daemon --dry-run` beside it.
   - Check χ²/dof per channel (expected 1.1–2 in clean channels, section 3.6), the number of power-flagged inputs, the Sun scale, and the stability of the tracked delays over a day.
3. **Sign check on NDP data.** The convention was confirmed on ADP-era TBF data. The NDP F-engine must be checked the same way: the as-is and conjugated covariance must give clearly different χ² (section 3.6). This needs only the dry run.
4. **Apply.**
   - Add `--gain_table /dev/shm/epic_gains_gpu{gpu}` and drop `--dry-run`.
   - Compare A-team peak S/N and image rms before and after, and against the forward-modelled EPIC image of the model sky with the applied gains. This is the image monitor: `epicop.image_from_vis(op, W M W^H)` predicts the GPU image, and the ratio of A-team peaks gives a per-image flux scale in the manner of Kuiack et al. 2019.
5. **Six bands.** Repeat on all workers. The lowest bands (16.1 and 20.7 MHz) need the most attention to RFI gating and to ionospheric source shifts.

## 4.7 Open items

- **GPU-level test of gain application** in `tests/gpu/test_imager.cu`. A table with a linear phase ramp across the stands should shift a simulated source by the predicted number of pixels; a uniform |c| should scale the image by |c|². The Python operator reproduces the kernel exactly, but the CUDA path itself has not been run with a table.
- **Cross-hand (X–Y) phase.** It is left at the cable model by the gauge choice. Calibrating it needs the XY covariance, which the X-engine can produce at about one third extra cost, and a polarised model or the known zero-polarisation sky. The Stokes I images (XX+YY) do not depend on it.
- **Direction-dependent ionospheric shifts.** Add per-source position offsets as nuisance parameters for Cyg A and Cas A, as AARTFAAC does. The simulations include 0.1° shifts, which the direction-independent solution absorbs at the 1–2° level.
- **Sky model.**
  - **Done:** the A-team spectra of Cas A, Cyg A and Vir A come from de Gasperin et al. 2020. The integrated spectra are summed from their LOFAR LBA clean components (CDS J/A+A/635/A150, epoch 2015-08-26) into `streamcal/data/ateam_degasperin2020.csv`. Cas A fades at an assumed 0.7% per year. On the 14 night TBF captures, the fitted Cas A / Cyg A flux-scale ratio is 0.96 (IQR 0.82–1.19), against 1.70 with the Baars spectra. Both sources then need the same scale, about 0.8, at all elevations, which is the A-team relative to the diffuse map: a diffuse-map or beam calibration, not an A-team error. Tau A fits about 1.5× the de Gasperin value and keeps the earlier model. Below 30 MHz the spectra are extrapolated, and Cas A's low-frequency turnover is not modelled; check this before using them for the 16 and 21 MHz bands. Choice via `STREAMCAL_ATEAM`.
  - **Tested, no measurable gain yet:** 120–400 VLSSr sources above 20 Jy at 74 MHz (merged within 0.5°; `streamcal/data/vlssr_bright.csv`, `STREAMCAL_CATALOG=vlssr`), added as fixed flux. χ²/dof (1.59) and the residual fraction (0.1825) are unchanged, because the diffuse map and the beam dominate the residual. The option stays off by default.
  - Both tables are built by `research/build_sky_catalogs.py`; `research/sky_eval.py` scores a sky model on the TBF captures.
  - **Diffuse map default `auto`** (section 2.4). The OVRO-LWA hybrid is used at 30–80 MHz and LFSM outside that range (the 16 and 21 MHz bands) or when a node has no hybrid file.
    - The hybrid now uses all eight release maps (36.5–73.2 MHz; `ovrolwa8_hybrid_n256.npz`, 28 MB), so it covers the 57, 72 and 76 MHz bands too.
    - It is installed on every worker's local disk: `/data/streamcal/diffuse_maps/` on all workers except epic5, which uses `/var/tmp/streamcal/diffuse_maps/`. `sky.OVROLWA_MAPS` lists both locations, and `$STREAMCAL_OVROLWA_MAP` overrides them. The archived copy is in `/FileStore/epicT/epic/streamcal_archive/`.
    - On the TBF captures `auto` reproduces the OVRO-LWA result: χ²/dof 1.48, residual 0.1766.
  - **Catalogue and OVRO-LWA together.** The OVRO-LWA release removed only its eight brightest sources. With `STREAMCAL_CATALOG=vlssr`, the catalogue sources are inpainted out of the map (r 0.75°, annulus 0.75–1.25°), so they are not counted twice. The combination is neutral on the TBF captures (residual 0.1766–0.1767 for limits of 20 and 50 Jy).
  - **A-team below 30 MHz.** The de Gasperin data end at 30 MHz. Below that, the Baars spectral shapes are scaled to match at 30 MHz: Cas A 49 kJy at 20 MHz and 58 kJy at 16 MHz. These remain uncertain (the Baars Cas A fit is formally valid above 300 MHz). The daemon solves the A-team fluxes freely within 0.25–4× the catalogue, so the catalogue only sets the starting point for the 16 and 21 MHz bands.
  - Cache model matrices on an LST grid for every band.
- **Tracker tuning.** The process noise ($`2^\circ/\sqrt{\rm s}`$, $`0.05\ \mathrm{ns}/\sqrt{\rm s}`$) was chosen for simulated drifts. The measured drifts (section 3.6) are about 10× slower, so $`Q`$ can be lowered and the effective integration lengthened.
- **Gridding correction.** `gridding_correction` takes its live-stand mask from non-zero kernels, not from flags. Stands flagged by calibration remain in the correction's mean kernel, which is a small, smooth image-plane effect.
- **Prometheus metrics.** Export χ², flags, Sun scale, gain version and tap `nskipped` from the daemon. The image metadata already carries `gain_version`. The FITS writer should copy it into a `GAINVER` header keyword.
