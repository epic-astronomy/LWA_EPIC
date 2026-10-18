# 7. Imager and calibrator benchmark

All measurements were made on epic1 on 2026-10-17:

- **Hardware:** RTX 4090 at default clocks (production locks them higher); Threadripper PRO 5965WX, 24 cores.
- **Imager:** the branch build, run in its benchmark mode (`epic++ --bench`, section 4.2). The graph is generator → `CorrelatorRft` (with the calibration tap and gain table) → sink, with no packet capture, database, disk or video.
- **Imaging options:** production, i.e. `--imagesize 128 --imageres 1.056 --nts 1000 --seq_accum 80 --channels 128 --support 5 --aeff 25 --accum_16bit --nstreams 8 --kernel_oversample 4 --wproj --remove_autocorr`.
- **Environment:** the `epic310` conda environment, as the Nomad launcher uses.
- **Calibrator:** `streamcal.daemon` on the same node, pinned to cores 12–23 with 8 worker processes and 8 BLAS threads.

## 7.1 Simulated F-engine data with wrong gains

The dataset is built in two steps:

- `research/make_bench_model.py` builds, for all 128 channels of the 75.8 MHz band and both polarisations, the true covariance of all 256 stands. The sky is LFSM at nside 128 with 10% small-scale structure, plus 25 faint sources, the A-team and the Sun. This truth deliberately differs from the daemon's nside 64 model.
- `research/make_bench_gulps.py` draws correlated voltages from it, quantises them to 4+4 bits (rms 2 levels per component at the median input power, clipped at ±7) and writes 96 gulps (6.3 GB, 4 s) to `/dev/shm`.

The gains are wrong in the raw F-engine convention, so that epic++'s nominal phases (geometry × LSL cable model) leave a residual of:

- amplitude scatter 20%;
- phase offsets uniform within ±70°;
- residual delays of 3 ns rms;
- three dead inputs (2% gain).

The generator plays the 96 gulps cyclically. On each pass it rotates every channel's sequences by 50, so the 5% tap sees new samples on every pass.

## 7.2 Timing

| Configuration (unpaced: gulps as fast as the imager accepts them) | Interval per gulp, mean / p99 / max | Images: interval, mean / p99 |
|---|---|---|
| Empty gulps, no calibration | 35.5 / 35.9 / 36.1 ms | 71.0 / 76 ms |
| Empty gulps, tap at 50 sequences per gulp, batch 1 | 35.4 / 35.8 ms (tap kept 121 of 243 gulps) | 70.8 / 73 ms |
| Empty gulps, tap at 50 sequences per gulp, batch 8 | 35.4 / 35.6 / 35.7 ms (tap kept 240 of 240) | 70.8 / 72.6 ms |
| Simulated data, exact (oracle) gain table | 35.6 / 36.5 / 36.7 ms | 71.2 / 76 ms |
| Simulated data, tap, daemon solving, tables applied | 35.8 / 36.4–36.8 / 37.3 ms | 71.7 / 76.4 ms |

A gulp holds 41.8 ms of data, and an image (two gulps) 83.6 ms. With the full calibration loop the imager uses 86% of real time at the p99, leaving 5 ms of headroom per gulp. The calibration adds 0.2–0.3 ms per gulp:

- staging the tap slice;
- the 512 kB phase upload when a table changes;
- CPU memory traffic shared with the X-engine and the daemon.

**Real time.** With the generator paced at the F-engine rate, the imager never fell behind over 7,200 gulps (5 min). The generator waited 0.4 µs on average for a free buffer, and images arrived every 83.59 ms on average. The tap correlated every gulp.

**Daemon.** One table covers 32 solved channels × 2 polarisations, with sky fluxes and positions. A cold solve takes 3.9 s and a warm one 1.5–1.7 s; a model refresh takes 6.1 s per minute. The first version solved serially, took 16–17 s and skipped every other table.

## 7.3 Calibration of the simulated data through the imager

`research/bench_watch.py` scores every gain table the daemon publishes against the true gains of the dataset, over all 128 channels. A table is scored as the rms phase of $`c\,\varepsilon`$ with tip/tilt removed, plus the image shift and the amplitude spread. `research/bench_images.py` compares the GPU images, dumped every 30 images, with images made under the exact gain table.

| Gain table (10 s each) | Phase X / Y | Image shift | Amplitude X / Y | Dead inputs | GPU image residual against exact gains | Cyg A peak / exact |
|---|---|---|---|---|---|---|
| none (cable model) | about 40° | | 20% | in use | 0.219 | 0.970 |
| v1 (first 10 s) | 0.70° / 0.71° | 0.01° | 4.7% / 5.6% | flagged | 0.029 | 1.013 |
| v3 | 0.61° / 0.62° | 0.01° | 3.9% / 4.9% | flagged | 0.019 | 1.013 |
| v10–v29 | 0.56–0.65° | 0.01–0.02° | 3.0% / 4.2% | flagged | 0.019 | 1.013–1.015 |

The residual is at the sky-model floor: section 3.2 found about 0.4–0.5° and 1–2% for the same kind of model error at 75.8 MHz. Every gain table was applied at the next image boundary (29 in 5 minutes), and every image carries its `gain_version`.

## 7.4 Defects found by the benchmark

1. **Buffer race.** The first tap run crashed with a segfault in `BenchGulpGen` writing gulp metadata. `Payload::unlock` released the buffer before `ResetBuf()` cleared its metadata, so a producer spinning on `acquire_buf()` wrote into a map being cleared. This was fixed in `buffer.hpp`; the production packet assembler is exposed to the same race.
2. **Tap throughput.** With one gulp per X-engine call, the read-modify-write of the 134 MB accumulator kept the tap to every second gulp. Batching eight gulps per call fixed it.
3. **Absolute gain scale.** Against a model in Jy, the solved gains are about 0.01 (F-engine counts per Jy^½). Their inverses were clipped at the writer's limit, so the published table had no amplitude correction at all. Once it reached the GPU, the factor of 32 overflowed the half-precision gridding/FFT and every image became NaN. The fix: the daemon publishes relative corrections (median |c| = 1 per channel and polarisation). Both the Python writer and the epic++ reader reject tables whose median non-zero |c| lies outside 0.25–4.
4. **Kalman time step from model time.** Under `--fixed-time` the time step collapsed to the 1 s floor. The fix takes it from the data timestamps.
5. **Cold start.** The first solve fitted the sky parameters with unknown gains, which gave A-team fluxes near zero and 4–6° offsets. The fix solves the gains against the nominal sky before freeing the sky parameters.
6. **Startup warning.** The gain file existed with an invalid header until the first publish, which produced "bad header" warnings. The writer stamps a valid, inactive header when it creates the file.

## 7.5 Reproduction

```bash
# step 1, any host with pygdsm and lsl (about 10 min)
python src/python/streamcal/research/make_bench_model.py /FileStore/epicT/epic/streamcal_bench/model75 --chan0 3104
# step 2, on the GPU node (numpy only; about 15 min for 96 gulps)
python src/python/streamcal/research/make_bench_gulps.py /FileStore/epicT/epic/streamcal_bench/model75 \
       /dev/shm/streamcal_bench/gulps.bin --ngulps 96
# reference images with the exact gains
ORACLE=1 DATA=/dev/shm/streamcal_bench/gulps.bin src/python/streamcal/research/bench_run.sh oracle 480 \
       --bench_dump /dev/shm/streamcal_bench/oracle/dumps --bench_dump_every 8
# imager + tap + streamcal in real time
CAL=1 DATA=/dev/shm/streamcal_bench/gulps.bin src/python/streamcal/research/bench_run.sh cal_rt 7200 \
       --bench_realtime --bench_dump /dev/shm/streamcal_bench/cal_rt/dumps --bench_dump_every 30
python src/python/streamcal/research/bench_images.py /dev/shm/streamcal_bench/cal_rt/dumps /dev/shm/streamcal_bench/oracle/dumps
```

`bench_run.sh` expects the daemon's dependencies in `/FileStore/epicT/epic/streamcal_bench/venv`. That is a venv over `epic310` (`--system-site-packages`) with `pygdsm` added, so the production environment is unchanged.
