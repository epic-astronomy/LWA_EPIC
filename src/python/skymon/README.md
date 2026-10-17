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
   - **Interference flagging** (`skymon_normalise` in `_kernels.c`). After the median, the spread of each channel over the sky (1.4826 × the median absolute deviation over a subset of up to 2048 pixels) is about 1 for noise. A narrowband carrier seen over much of the sky (aircraft or meteor scatter of a distant transmitter) widens it, while a burst, confined to a beam, does not. A channel whose spread exceeds `chan_flag_sigma` (1.6) times that of the band (the median spread over the channels, or 1 if smaller) is set to zero for every pixel in that image. A rise of noise over the whole band (a thunderstorm, a solar burst, a change of gain) widens every channel alike; it is not flagged, but searched and recorded in the conditions (step 8).
   - A channel flagged persistently is masked: its flag rate (exponential, 30 s) above 0.5 sets it to zero in every image until the rate falls below 0.2.
   - A single cell (pixel and channel) beyond `clip` (12) is set to zero. A burst of S/N s adds about s / sqrt(nchan × samples per channel) to a cell, under 9 for s = 100 with 128 channels, so the clip removes only what no burst produces.
   - Flagged channels and clipped cells do not update the running statistics. A cell withheld for 30 s in a row is updated again (deviations limited to 5σ), so a lasting change of level is followed in the end.
   - `/api/status` reports `rfi_flagging`: the fraction of channels zeroed over the last minute, the masked channels and the flags per channel. Each candidate's `quality.flagged_fraction` is the fraction zeroed along its sweep.
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
   - A candidate whose signal lies in a few channels is classified as RFI (`narrowband (a few channels)`), with its data kept. Along the dispersion track, each channel's samples are summed over the boxcar at that channel's delay (`quality.channel_contributions`). From these contributions c (positive part), `n_eff_channels` = (Σc)² / Σc² and `top2_fraction` is the share of the two strongest channels. Below `--narrowband-min-channels` (6) effective channels, or above 0.6 in two channels, the candidate is narrowband. A dispersed burst spreads over most of the band, while the carrier of an aircraft or of meteor scatter occupies one or two channels of 24 kHz.
   - A candidate at a known source is stored, with its data, as class `known` (`known.py`), and its `known_source` field gives the kind, name and separation:
     - `solar`, `jovian`: within 3 beams of the Sun or Jupiter. The positions come from astropy's built-in ephemeris for the candidate's time, are converted to FK5 J2000 like the image WCS, and are cached for 60 s.
     - `bright_source`: within 1.5 beams of Cyg A, Cas A, Tau A, Vir A, Her A or Hya A.
     - `pulsar`: within 2 beams of a bright low-frequency pulsar, with |DM − DM_psr| ≤ max(3 DM steps, 10%). The list is built in, and a CSV (`--known-pulsars`, columns name,ra_deg,dec_deg,dm,period_s) extends it or replaces entries of the same name.
     - The beam is that at the band centre, widened towards the horizon like the clustering radius.
   - A burst found again by another tier or block is stored once.
6. **Products.** Each candidate is a JSON record plus an `.npz` file under `<out>/candidates/YYYYMMDD/`. The `.npz` holds:
   - the pixel's dynamic spectrum at the tier's resolution;
   - its DM–time plane;
   - 30 min of the raw accumulated spectrum;
   - the noise table.
7. **Accumulation.** The raw spectrum of every tracked pixel, averaged over 10 s, is kept for 6 h in float16. This serves the spectrograms on the website.
8. **Conditions** (`conditions.py`). Thunderstorms, solar activity and interference episodes raise the noise or the trigger rate for minutes to hours. Every 10 s of data (on the image clock) the monitor records a sample:
   - `noise_ratio`: the median over tracked pixels and channels of the running variance over its slow baseline (exponential average, 1 h; it starts at 1);
   - `rfi_rate` and `cand_rate`: RFI-class and all candidates per minute, over the last minute;
   - `sky_fraction_triggered`: the fraction of tracked pixels with a hit in the search blocks of the interval;
   - `missing_fraction`: the fraction of missing images;
   - `disturbed`: noise ratio above 1.5, RFI rate above `--disturbed-rfi-rate` (2 per minute), or more than 1% of the sky triggered.

   The last 6 h are kept in memory (`/api/status`, `/api/conditions`) and written to the table `conditions` of the node's database by the database thread; samples wait in memory while the database is away. Each candidate carries in `quality.conditions` the summary of the minute before its arrival, and each decided event of the combiner carries the conditions of the bands that detected or measured it, with `disturbed` set when any of them was disturbed.

9. **Calibration.** With `epic++ --gain_table` the images are calibrated by streamcal (branch streaming-calibration). Each image carries the version of the gain table it was made with (slot header offset 56 of the ring; 0: nominal phases).
   - The monitor counts the changes of version in the conditions (`gain_changes`). It logs when calibration is first applied, and when it reverts to the nominal phases.
   - Each candidate records in `quality.calibration` the table version and the changes within its sweep. `applied_or_reverted_in_sweep` marks the larger change of the images when calibration starts or stops. No image is held back.
10. **Flux and fluence** (`flux.py`). The gain tables are relative (median |c| = 1 per channel) and epic++ applies no primary beam, so the images have no absolute scale. With `--calinfo /dev/shm/epic_calinfo_gpu0.json`, the monitor reads the calibration record streamcal writes with every solution. The record holds:
    - the absolute scale F_X, F_Y per channel;
    - the calibrators, with their model spectra, solved scales, positions and dipole responses;
    - a table of the dipole response.

    Every minute the monitor measures each calibrator above 20° in its own accumulation, as the excess of its pixel over a ring 2–4 beams around it. It divides the excess by S (F_X² B_X + F_Y² B_Y) / 2 to get k, in image units per Jy, per channel. A candidate's fluence per channel is

    sum along the track (normalised) × σ_raw × sqrt(2^tier) × t_image / (k Beff(position)),

    and `flux` gives it in Jy ms over the band and per channel, with the mean flux density over the boxcar, the beam relative to the zenith and the calibrators used. The `.npz` adds:
    - `raw_sigma` and `raw_mean`, the pixel's noise and mean per image in image units;
    - `tier_images` and `image_s`;
    - the calibration record and the calibrator measurement as JSON.

    Records and measurements go to the table `calibration` of the node's database. On a simulated sky with its own record (`simtest` writes one), Cyg A gave k = 0.98 of the true value. A burst of known fluence was measured at 0.82 of it, about the same loss as its S/N (34 found for 40 injected): the burst lies off the centre of its HEALPix pixel, and the search filters remove part of it.

## Consensus across bands (combiner.py)

One combiner, on any imaging node, reads the candidates of every monitor every 10 s. A real burst shows the same position and DM in every band that can see it, at times t_bottom = t_inf + K DM / f_lo^2, up to hours apart between the bands.

For each new candidate the combiner opens an event, or joins the candidate to an open event that it matches in position, DM and arrival time.

A candidate at the event's position whose arrival falls within the event's sweep through its band, but at a clearly different DM, is listed in the event's `other_dm`, with a note. It may be the same burst found at a wrong trial DM, or a second burst from the same place (a repeater). Its data stay linked to the event and are archived with it. It does not replace the band's detection or change the fitted DM. The reviewer decides which it is.

For every other band, the combiner then:
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

Monitors keep the data of RFI candidates too, unless the RFI is certain (DM < 2, or much of the sky at once). Each candidate also carries in-band quality measures (`quality.py`). Only class `candidate` (and coordinated injections) opens events; candidates at known sources (`known`) do not. On the node, the archiver (`archiver.py`) removes `known` candidates that are not archived after `--review-days` (14).

**Cross-band coincidence veto.** Terrestrial impulsive interference (lightning sferics, meteor afterglows, sparks) is undispersed: it reaches every band at the same moment, whereas a burst from the sky reaches the bands seconds to hours apart.
- The combiner keeps the new candidates and RFI of every node for 15 min (`coinc_history_s`).
- Two candidates from different nodes are simultaneous when |t_bottom_a − t_bottom_b| ≤ 1.5 s + tier_dt_a + tier_dt_b + width_a + width_b.
- A simultaneous pair is undispersed when the dispersion law at either candidate's DM predicts a gap between the two bands larger than 3 s plus the same tolerances. A burst at that DM could not reach both bands at once.
- Positions are not compared, because interference in the near field is imaged at different places in different bands.
- Bands close in frequency, or a low DM, predict a gap too small to separate the two cases. Such pairs are left to the consensus.

The candidates of all bands involved form one event of kind and status `terrestrial` (`review: not needed`), with their node, id, t_bottom, DM and S/N, and the reason. A coincident candidate opens no sky event. A pending sky event whose detections are all coincident is given status `terrestrial` with a note; the record is kept. `/api/status` counts these events under `events.terrestrial` and `terrestrial`.

Events go to `<out>/events/YYYYMMDD/*.json` and are served on :8096 (`/api/status`, `/api/events`).

**External context** (`context.py`). Solar radio bursts and nearby thunderstorms produce false events. When an event is decided or reopened, the combiner attaches `context` for the reviewer:
- `sun_el_deg` and `sun_sep_deg`: the Sun's elevation at LWA-SV at the first arrival, and its angle from the event's position (astropy, without downloads);
- `solar`: NOAA SWPC reports within ±15 min of the arrival in any band. These are radio sweeps (`RSP`, types II, III, IV, ...) overlapping the band in frequency, and X-ray flares (`XRA` and the GOES flare lists), each with its class and its begin, maximum and end times (UTC);
- `lightning`: strikes within ±10 min of the arrivals, as `strikes_100km` and `nearest_km` (with `complete` when the data cover the whole window);
- `sources` lists the providers with data; `unavailable` lists those that are enabled but have none.

Each provider is polled in its own thread every `--context-every` minutes (15). The last good data are kept in memory and in `<out>/context/`. A failure is logged once and does not affect the combiner. The context of an event decided in the last 24 h is updated when new reports arrive, since SWPC edits its lists hours after the events. SWPC is on by default (`--no-context-swpc` turns it off). Lightning is off unless `--lightning-url` gives a URL or a local file of JSON strikes, `{"strikes": [{"time": <Unix s or ISO 8601>, "lat": <deg>, "lon": <deg>}, ...]}`.

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
| `GET /api/conditions?since=` | data-quality samples (10 s, last 6 h), thresholds and summary |
| `GET /api/status` → `calibration` | gain table version, changes in the last hour, the latest calibration record and calibrator measurement |
| `POST /api/watch` | measure a position, DM and arrival time (from the combiner) |
| `POST /api/inject` | inject a burst (the combiner's coordinated injections) |

## Tests

```bash
cd tests/python && python -m unittest -v test_known test_chanflag test_skymon test_combiner test_coincidence test_context test_conditions test_localdb test_flux
```

The tests cover:
- the FDMT against numpy and brute force;
- tier coverage;
- the shared-memory ring, including overruns and writer restarts;
- the C++ publisher's pixel order against `epic_utils.save_output`;
- the geometry;
- detection by the searcher;
- interference flagging (`test_chanflag.py`): one channel and one image, noise without flags, the persistent mask and its release, and a moving carrier in a simulated sky that leaves no candidate while a burst is found;
- an end-to-end simulated sky with stored candidates and the HTTP interface;
- live injections;
- known-source labels (Sun, pulsars, bright sources) and the narrowband classification (`test_known`);
- a monitor that waits for a missing imager and survives a restart and a gap;
- the combiner's consensus, and the external context (parsing of saved SWPC samples in `tests/python/data/`, matching, lightning, unreachable sources);
- conditions during a minute of threefold noise, over HTTP, in the combiner's events, and in the database through an outage.
