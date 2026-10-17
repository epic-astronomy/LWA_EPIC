# skymon: EPIC all-sky monitor

`skymon` searches the whole sky above the elevation limit, in one imaging band, for dispersed radio bursts in real time. It reads the images that `epic++` publishes to shared memory on the same node.

Each imaging worker runs its own monitor for its own band, with no dependency on other workers, the database or the shared file system. A worker that goes down removes only its band.

```
epic++ (ChanReducer) ──Stokes I──▶ /dev/shm/<ring> ──▶ skymon run ──▶ <out>/candidates/YYYYMMDD/*.json, *.npz
                                                          │            <out>/injections.jsonl
                                                          └──HTTP :8095 (status, candidates, pixel spectrograms)
```

## Data path

1. **Publishing.** `epic++ --sky_shm <name>` writes every image's XX + YY, at full channel resolution, to a ring of slots in `/dev/shm/<name>`.
   - The code is `src/ex/sky_publisher.hpp`, called from ChanReducer.
   - The pixel order is the same as the saved FITS images, and the layout is documented in `shm.py`.
   - The writer never waits. A reader that falls behind loses the oldest images, and counts them.
2. **Sky-fixed pixels.** The monitor tracks HEALPix pixels (J2000, RING order) rather than image pixels, so a pixel's time series always looks at the same patch of sky.
   - Every 10 s it recomputes which pixels are above the elevation limit. Pixels that rise get a slot; pixels that set give theirs up.
   - Each tracked pixel's position on the image is recomputed with the imager's own WCS (`epic_geometry.apply_wcs`).
   - Each image is interpolated onto the tracked pixels by bicubic convolution (16 taps), which keeps the ripple of a bright source between image pixels to about 2%.
   - The default `nside` gives pixels of at most half the beam (λ/D): nside 64 above 40 MHz, 32 near 35 MHz, 16 at 16–21 MHz.
3. **Normalisation.** The values of every pixel and channel are normalised by a running mean (2 min) and variance (5 min). The variance comes from successive differences, so a source rising through the primary beam does not inflate it. Outliers are clipped out of these statistics.
   - The median over the sky of each channel is subtracted, which removes RFI and gain changes common to the whole image.
4. **Dedispersion in tiers** (`search.py`, `dedisp.py`).
   - Tier j averages 2^j images and runs the fast dispersion measure transform (FDMT) over delays 0..D samples, with D equal to the number of channels.
   - The dispersion delay is linear in DM, so tier j covers DMs 2^j times those of tier 0. The tiers together cover 0–`--dm-max` in octaves.
   - At the top of each tier, a pulse is smeared by about one sample per channel, so averaging further would lose signal.
   - Every tier keeps only D + block samples of history, whatever its DM. All tiers together take about 0.7 MB per pixel for 128 channels (14 GB at nside 64), and the accumulation (step 7) another 12 GB.
5. **Search.**
   - A straight line fitted over the window is removed from every channel of every pixel before the FDMT. Above tier 0 the mean over channels at each time is removed too (zero-DM filter), which takes out broadband variations at a cost of 1–2 parts in nchan of a burst.
   - Each tier searches boxcars of 1, 2, 4 and 8 samples. The noise of every delay row and width is measured from the data (clipped RMS of the previous blocks).
   - Hits above `--threshold` (S/N 8) are clustered over the sky, DM and time.
   - Two kinds of cluster are classified as RFI: one wider than six beam areas, and one at a moment when more than 1% of the sky triggers.
   - A burst found again by another tier or block is stored once.
6. **Products.** Each candidate is a JSON record plus an `.npz` file under `<out>/candidates/YYYYMMDD/`. The `.npz` holds:
   - the pixel's dynamic spectrum at the tier's resolution;
   - its DM–time plane;
   - 30 min of the raw accumulated spectrum;
   - the noise table.
7. **Accumulation.** The raw spectrum of every tracked pixel, averaged over 10 s, is kept for 6 h in float16. This serves the spectrograms on the website.

## Consensus across bands (combiner.py)

One combiner, on any imaging node, reads the candidates of every monitor every 10 s. A real burst shows the same position and DM in every band that can see it, at times t_bottom = t_inf + K DM / f_lo^2, up to hours apart between the bands.

For each new candidate the combiner opens an event, or joins the candidate to an open event that it matches. For every other band, it then:
1. predicts the arrival and its time tolerance (the DM uncertainty extrapolated across the bands);
2. checks the position is above that monitor's elevation limit for the whole sweep;
3. sends `POST /api/watch`.

The monitor measures the watch when the tier for that DM reaches the predicted time (`watch.py`). It takes the S/N whatever its value, over the position's pixel and neighbours, within the DM and time tolerances. The result is stored with the pixel's data as a `followup` candidate.

With detections in two or more bands, DM and t_inf are refitted to all arrival times.

Each band then gives a value for the event: the S/N of a detection or followup, or an upper limit where only that band's own search covered the arrival. `spectrum.py` fits these values with power laws, band-limited (Gaussian) spectra, and scattering. Once every band is resolved, the event is classified:
- `confirmed`: two or more bands detect it, or measure it at S/N ≥ 6;
- `band_limited`: one band sees it, and the other bands fit a spectrum a burst can have;
- `implausible`: one band sees it, and no such spectrum fits (in practice, a contradiction between nearly overlapping bands);
- `unconfirmable`: no other band could see it (not visible, monitor down, or DM beyond its search).

No result discards data:
- every band keeps its detection, followup, or context record (the accumulated spectrum around the arrival);
- every sky event is marked `review: pending`;
- a verdict is recorded with `POST /api/events/<id>/review {verdict: frb|rfi|unclear, by, note}`.

Monitors keep the data of RFI candidates too, unless the RFI is certain (DM < 2, or much of the sky at once). Each candidate also carries in-band quality measures (`quality.py`).

**Cross-band coincidence veto.** Terrestrial impulsive interference (lightning sferics, meteor afterglows, sparks) is undispersed: it reaches every band at the same moment, whereas a burst from the sky reaches the bands seconds to hours apart.
- The combiner keeps the new candidates and RFI of every node for 15 min (`coinc_history_s`).
- Two candidates from different nodes are simultaneous when |t_bottom_a − t_bottom_b| ≤ 1.5 s + tier_dt_a + tier_dt_b + width_a + width_b.
- A simultaneous pair is undispersed when the dispersion law at either candidate's DM predicts a gap between the two bands larger than 3 s plus the same tolerances. A burst at that DM could not reach both bands at once.
- Positions are not compared, because interference in the near field is imaged at different places in different bands.
- Bands close in frequency, or a low DM, predict a gap too small to separate the two cases. Such pairs are left to the consensus.

The candidates of all bands involved form one event of kind and status `terrestrial` (`review: not needed`), with their node, id, t_bottom, DM and S/N, and the reason. A coincident candidate opens no sky event. A pending sky event whose detections are all coincident is given status `terrestrial` with a note; the record is kept. `/api/status` counts these events under `events.terrestrial` and `terrestrial`.

Events go to `<out>/events/YYYYMMDD/*.json` and are served on :8096 (`/api/status`, `/api/events`).

**Coordinated injections.** The combiner sends `POST /api/inject` with the same burst to every monitor, which checks the consensus end to end.

**Independence.** The monitors never depend on the combiner. The combiner keeps its state on disk, so it can be restarted on any node.

```bash
python -m skymon combine --nodes epic1=http://epic1:8095,...,epic6=http://epic6:8095 --out /FileStore/epic_skymon/combiner --port 8096 --inject-every 6
python -m skymon frblist --out frbs.json --start $(date +%s) --duration 3600   # the same FRBs for every band's simulator
python -m skymon simtest --realtime --frb-file frbs.json --chan0 1378 ...     # one per band, on its own worker
```

`POST` requests need the header `X-Skymon-Token` when `SKYMON_TOKEN` is set.

## Live test injections

Every `--inject-every` minutes on average (Poisson), the monitor adds a simulated dispersed burst to the incoming image, before regridding.
- **Where.** A random tracked position that stays up for the whole sweep.
- **DM.** Log-uniform over what that position can still complete.
- **Strength.** A target S/N of 10–30, set from the measured noise of that pixel.

A candidate that matches an injection is classified as `injection`. Every injection is logged twice in `<out>/injections.jsonl`: when it is scheduled, and again as recovered or missed. The detection efficiency of the running system is therefore measured continuously; `/api/status` reports the totals.

## Use

The `epic310` environment has everything the monitor needs: numpy, astropy, healpy, lsl and gcc. The compiled kernels (`_kernels.c`) are built on first use into `~/.cache/skymon/<host>/`.

```bash
export PYTHONPATH=/path/to/LWA_EPIC/src/python
python -m skymon run --shm epic_sky0 --out /data/skymon --port 8095 --threads 8
python -m skymon simtest --duration 1800 --realtime --out /tmp/simtest     # simulated sky, then evaluation
python -m skymon simulate --shm skysim --duration 3600                    # images only, at the real rate
python -m skymon evaluate --truth /tmp/simtest/truth.jsonl --out /tmp/simtest
```

`epic++` needs `--sky_shm epic_sky0` to publish. The ring holds 64 images by default (5.3 s, 537 MB for 128 channels).

## HTTP interface

| Request | Answer |
|---|---|
| `GET /api/status` | band, tiers, tracked pixels, real-time load, lag, candidate counts, injection statistics |
| `GET /api/pixel?ra=&dec=` (or `hpix=`, or image `x=&y=`), `&since=`, `&fine=1`, `&format=npz` | spectrogram of the tracked pixel at a position: accumulation by default, the latest tier-0 window with `fine=1` |
| `GET /api/candidates?since=&cls=&limit=` | recent candidates |
| `GET /api/candidates/<id>`, `/api/candidates/<id>/npz` | one candidate and its data products |
| `GET /api/sky` | tracked pixels with their age and noise |
| `GET /api/injections` | injection summary |
| `GET /api/watches?event=` | watches requested by the combiner, and their results |
| `POST /api/watch` | measure a position, DM and arrival time (from the combiner) |
| `POST /api/inject` | inject a burst (the combiner's coordinated injections) |

## Tests

```bash
cd tests/python && python -m unittest -v test_skymon
```

The tests cover:
- the FDMT against numpy and brute force;
- tier coverage;
- the shared-memory ring, including overruns and writer restarts;
- the C++ publisher's pixel order against `epic_utils.save_output`;
- the geometry;
- detection by the searcher;
- an end-to-end simulated sky with stored candidates and the HTTP interface;
- live injections;
- a monitor that waits for a missing imager and survives a restart and a gap.
