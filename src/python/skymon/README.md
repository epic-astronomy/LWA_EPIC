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
   - Each image is interpolated onto the tracked pixels by bicubic convolution (16 taps, Keys a = −0.75). On the beam of the images, about three image pixels per FWHM, it reproduces a peak to between 0.0% and +0.8% (+0.4% on average). Keys' usual a = −0.5 lost 2.5% on average (up to 4.8%), and bilinear interpolation 10%.
   - The default `nside` gives pixels of at most half the beam (λ/D): nside 64 above 40 MHz, 32 near 35 MHz, 16 at 16–21 MHz.
3. **Normalisation.** The values of every pixel and channel are normalised by a running mean (2 min) and variance (5 min). The variance comes from successive differences, so a source rising through the primary beam does not inflate it. Outliers are clipped out of these statistics.
   - The median over the sky of each channel is subtracted, which removes RFI and gain changes common to the whole image.
   - **Interference flagging** (`skymon_normalise` in `_kernels.c`). After the median, the spread of each channel over the sky (1.4826 × the median absolute deviation over a subset of up to 2048 pixels) is about 1 for noise. A narrowband carrier seen over much of the sky (aircraft or meteor scatter of a distant transmitter) widens it, while a burst, confined to a beam, does not. A channel whose spread exceeds `chan_flag_sigma` (1.6) times that of the band (the median spread over the channels, or 1 if smaller) is set to zero for every pixel in that image. A rise of noise over the whole band (a thunderstorm, a solar burst, a change of gain) widens every channel alike; it is not flagged, but searched and recorded in the conditions (step 8).
   - A channel flagged persistently is masked: its flag rate (exponential, 30 s) above 0.5 sets it to zero in every image until the rate falls below 0.2.
   - A single cell (pixel and channel) beyond `clip` (12) is set to zero, unless it is sky-shaped: at least 3 of its 8 HEALPix neighbours deviate the same way, in the same channel and image, by at least 0.3 of its deviation (`clip_sky_neighbours`, `clip_sky_fraction`).
     - A burst whose signal in a channel falls into one image adds S/N × sqrt(n_eff / nchan) to that cell. Here n_eff, about 1.7 at the top of a tier, is the number of samples per channel in the S/N's definition. The clip is therefore reached at S/N 73 for 64 channels, 104 for 128 and 127 for 192.
     - Anything from the sky reaches the image through the synthesised beam, wider than a sky pixel, so a neighbour sees 0.5–0.8 of a point source's peak. A bright burst therefore keeps its cells, while a glitch in a single cell is still removed.
     - Before this exemption, a burst of S/N 120 in 64 channels lost its core pixels: it was found 1.5° away at S/N 54 with 0.45 of its fluence.
   - Flagged channels and cells beyond the clip, kept or not, do not update the running statistics. A cell withheld for 30 s in a row is updated again (deviations limited to 5σ), so a lasting change of level is followed in the end.
   - `/api/status` reports `rfi_flagging`: the fraction of channels zeroed over the last minute, the masked channels and the flags per channel. Each candidate's `quality.flagged_fraction` is the fraction zeroed along its sweep.
4. **Dedispersion in tiers** (`search.py`, `dedisp.py`).
   - Tier j averages 2^j images and runs the fast dispersion measure transform (FDMT) over delays 0..D samples, with D equal to the number of channels times `--max-delay-factor`.
   - The transform takes any number of channels; the number comes from the ring (`nchan`). It merges adjacent subbands in pairs; at a level with an odd number of subbands the lowest is passed on unmerged and joins one level later. The FX imager's 192 channels of 23.9 kHz (`--sky_nbin 1`) form three subbands of 64, of which the upper two are merged first. Nothing is padded, and the band edges are the real ones.
   - Each tier high-passes its input with a time constant of at least 32 of its samples (`--hp-tau`) and at least 4 times its widest boxcar (`--hp-factor`).
   - The dispersion delay is linear in DM, so tier j covers DMs 2^j times those of tier 0. The tiers together cover 0–`--dm-max` in octaves.
   - At the top of each tier, a pulse is smeared by about one sample per channel, so averaging further would lose signal.
   - Every tier keeps only D + block samples of history, whatever its DM. The histories are kept in half precision (`--history-dtype float16`, the default; every sum is formed in single precision, and a stored sample carries a relative rounding error of at most 2^-11). For 192 channels with D = 2C and a block of 512, all tiers together take 1.3 MB per pixel (26 GiB at nside 64, against 51 GiB in single precision), and the accumulation (step 7) another 16.5 GiB.
   - The FDMT and the boxcar search are compiled (`skymon_search` in `_kernels.c`). Each thread transforms a tile of 16 pixels depth first, so that the small subbands stay in the core's cache, and searches each final delay row as soon as it is formed.
   - Only the rows and samples that the boxcars read are formed. Above tier 0 only delays above D/2 are reported, and each subband forms only the rows its parent reads: about half of them. A final row is read from the first sample of the widest boxcar ending in the block on, and each subband is formed from the first sample its parent reads.
5. **Search.**
   - The mean over the window is removed from every channel of every pixel before the FDMT (`--detrend mean`; `linear` removes a fitted straight line, which takes up to 4w/W of a pulse of w samples at the end of a window of W). `--detrend-len` and `--detrend-factor` lengthen a tier's history beyond the transform's window, for the trend fit only. Above tier 0 the mean over channels at each time is removed too (zero-DM filter), which takes out broadband variations; a pulse of w samples loses about w/S of its signal, S being its sweep in tier samples, so a pulse longer than its sweep is lost in the higher tiers.
   - **Wide pulses.** FRBs at 57–76 MHz are broadened to seconds or minutes by scattering. Tiers 0–4 search boxcars of 1, 2, 4, …, 128 samples, and the tiers above 1, 2, 4 and 8 (`--widths`, one list or per tier range, e.g. `0-4:1,2,4,8,16,32,64,128;5-:1,2,4,8`). A boxcar of w samples is evaluated at every (w/8)-th end sample (`--boxcar-stride` 8), at an average cost of 1/32 of a matched pulse's S/N. Against the boxcars to 8 samples, this raises the FRB rate by a factor of 1.52–1.59 at 57–76 MHz for the reference population, at 0.004 of real time (docs/skymon/DESIGN.md, §3.3 and §10.1c).
   - The noise of every delay row and width is measured from the data (clipped RMS of the previous blocks).
   - Hits above the threshold are clustered over the sky, DM and time. `--threshold` (S/N 8) applies to the boxcars 1, 2, 4 and 8; the threshold applied is raised for the extra trials of the wider boxcars at the same Gaussian false-alarm rate, to 8.007 (`--no-far-match` to keep 8).
   - Two kinds of cluster are classified as RFI: one wider than six beam areas, and one at a moment when more than 1% of the sky triggers.
   - A candidate whose signal lies in a few channels is classified as RFI (`narrowband (a few channels)`), with its data kept. Along the dispersion track, each channel's samples are summed over the boxcar at that channel's delay (`quality.channel_contributions`). From these contributions c (positive part), `n_eff_channels` = (Σc)² / Σc² and `top2_fraction` is the share of the two strongest channels. Below `--narrowband-min-channels` (6) effective channels, or above 0.6 in two channels, the candidate is narrowband. A dispersed burst spreads over most of the band, while the carrier of an aircraft or of meteor scatter occupies one or two channels of 24 kHz.
   - A candidate at a known source is stored, with its data, as class `known` (`known.py`), and its `known_source` field gives the kind, name and separation:
     - `solar`, `jovian`: within 3 beams of the Sun or Jupiter. The positions come from astropy's built-in ephemeris for the candidate's time, are converted to FK5 J2000 like the image WCS, and are cached for 60 s.
     - `bright_source`: within 1.5 beams of Cyg A, Cas A, Tau A, Vir A, Her A or Hya A.
     - `pulsar`: within 2 beams of a bright low-frequency pulsar, with |DM − DM_psr| ≤ max(3 DM steps, 10%). The list is built in, and a CSV (`--known-pulsars`, columns name,ra_deg,dec_deg,dm,period_s) extends it or replaces entries of the same name.
     - The beam is that at the band centre, widened towards the horizon like the clustering radius.
   - A burst found again by another tier or block is stored once.
   - A bright burst also passes the threshold at trial DMs far from its own: off by ΔDM, the pulse is smeared over K ΔDM (f_lo⁻² − f_hi⁻²), and its S/N falls as the square root of that widening. These arms of its DM–time plane are not stored as candidates of their own. A weaker candidate within a beam of a brighter one, arriving within the brighter burst's smeared sweep, and with at most twice the S/N of such an arm (`dm_arm_of` in `candidates.py`), is listed in the brighter candidate's `dm_arms`; one stored before the brighter burst keeps its files, marked as superseded by it. Bursts of S/N 250 at DMs 30–162 each gave 4 to 12 stored candidates before this rule, and one each with it.
6. **Products.** Each candidate is a JSON record plus an `.npz` file under `<out>/candidates/YYYYMMDD/`. The `.npz` holds:
   - the pixel's dynamic spectrum at the tier's resolution;
   - its DM–time plane;
   - 30 min of the raw accumulated spectrum;
   - the noise table.

   The search window ends with the block that completed the burst, so at detection the burst sits at the right edge of its dynamic spectrum. The monitor keeps appending the pixel's samples from the following blocks of the same tier until as many follow the burst as its sweep spans (at least 32 samples, at most the transform's delay range), then writes the `.npz` again with the DM–time plane over the longer span (`Monitor._extend_tails`; at most 10 min, and it stops if the pixel sets). The data before the burst are limited to the search window: for a DM near the top of its tier, the window begins about when the burst reaches the band.
7. **Accumulation.** The raw spectrum of every tracked pixel, averaged over 10 s, is kept for 6 h in float16. This serves the spectrograms on the website.
8. **Conditions** (`conditions.py`). Thunderstorms, solar activity and interference episodes raise the noise or the trigger rate for minutes to hours. Every 10 s of data (on the image clock) the monitor records a sample:
   - `noise_ratio`: the median over tracked pixels and channels of the running variance over its slow baseline (exponential average, 1 h; it starts at 1);
   - `rfi_rate` and `cand_rate`: RFI-class and all candidates per minute, over the last minute;
   - `sky_fraction_triggered`: the fraction of tracked pixels with a hit in the search blocks of the interval;
   - `missing_fraction`: the fraction of missing images;
   - `search_noise`: the measured search noise relative to white noise through the same chain, at widths of about 1, 7 and 20 s (nearest tier and boxcar), the realised noise floor at the widths of scattered bursts; with `el_cyg_a` and `el_cas_a`, the elevations of Cyg A and Cas A;
   - `disturbed`: noise ratio above 1.5, RFI rate above `--disturbed-rfi-rate` (2 per minute), or more than 1% of the sky triggered.

   The last 6 h are kept in memory (`/api/status`, `/api/conditions`) and written to the table `conditions` of the node's database by the database thread; samples wait in memory while the database is away. Each candidate carries in `quality.conditions` the summary of the minute before its arrival, and each decided event of the combiner carries the conditions of the bands that detected or measured it, with `disturbed` set when any of them was disturbed.

9. **Calibration.** With `epic++ --gain_table` the images are calibrated by streamcal (branch streaming-calibration). Each image carries the version of the gain table it was made with (slot header offset 56 of the ring; 0: nominal phases).
   - The monitor counts the changes of version in the conditions (`gain_changes`). It logs when calibration is first applied, and when it reverts to the nominal phases.
   - Each candidate records in `quality.calibration` the table version and the changes within its sweep. `applied_or_reverted_in_sweep` marks the larger change of the images when calibration starts or stops. No image is held back.
10. **Flux and fluence** (`flux.py`). The gain tables are relative (median |c| = 1 per channel) and epic++ applies no primary beam, so the images have no absolute scale. With `--calinfo <file>`, the monitor reads the calibration record that the streaming calibrator writes with every gain table. What the record must hold, and why, is set out in `docs/skymon/streamcal-interface.md`:
    - the absolute scale F_X, F_Y per channel;
    - the calibrators, with their positions, model spectra and solved scales;
    - later, the beam calibrated after direction-dependent calibration.

    Until the record carries a beam, the monitor computes the LWA dipole response itself with lsl (`--beam-model`, default `empirical`). A beam table in the record takes precedence.

    Every minute the monitor measures each calibrator above 20° in its own accumulation, in image units:
    - **Pixel.** Of the calibrator's HEALPix pixel and its neighbours, the monitor takes the one that sees most of it. From the calibrator's known position, it computes that pixel's response R to the calibrator: the beam at the offset, through the pixel's interpolation taps.
    - **Background.** A quadratic surface is fitted to a ring 2–4 beams around the calibrator, with outliers rejected, and evaluated at the pixel. A median of the ring was biased by +5% of Cyg A on the curved background of the Galactic ridge in the simulated sky; the quadratic fit leaves +0.5%.
    - **k.** It is (pixel − background) / (R × S × (F_X² B_X + F_Y² B_Y) / 2), in image units per Jy, per channel.

    A candidate's fluence per channel is

    sum along the track (normalised, high-pass undone) × σ_raw × sqrt(2^tier) × t_image × P / (k Beff(position)).

    - **High-pass undone.** Each tier's high-pass leaves about −A / hp_tau on every sample after a pulse of amplitude A. Within each channel's span along the track, sample t is weighted by 1 + (samples of the span after t) / hp_tau, which inverts the high-pass exactly. Without it, a burst lost 7–9% of its fluence.
    - **P, the peak correction.** A Gaussian of the beam's width (λ / D) is fitted, in image coordinates, to the band-summed contributions of the candidate's pixel and its 8 neighbours. P is the fitted peak over the pixel's value, scaled per channel as the beam narrows with frequency. It is bounded to 0.8–1.3 and is 1 when a neighbour is not tracked. It also gives a fitted position. Followups get the same correction.

    `flux` gives the fluence in Jy ms over the band and per channel, with:
    - `fluence_err_jy_ms`, from the noise of each channel's contribution: its squared weights times the channel's robust noise per sample, measured in the window before the sweep. `fluence_err_method` says which method was used. Without the window, the error is 1.7 × fluence / S/N: the contributions span more samples than the boxcar of the S/N, and 1.66 / S/N was measured on simulated bursts;
    - `peak_correction`: the factor, the fitted position (`ra_deg`, `dec_deg`), and whether it was applied and why not. The same is in `quality.peak`;
    - the mean flux density over the boxcar;
    - the beam relative to the zenith, and the beam's source, model and version;
    - the calibrators used.

    The `.npz` adds:
    - `raw_sigma` and `raw_mean`;
    - `tier_images` and `image_s`;
    - the calibration record and the calibrator measurement.

    Records and measurements go to the table `calibration` of the node's database.

    **Simulated sky.** The test used a simulated sky with its own record (`simtest` writes one), at 57 MHz with 64 channels and nside 64.
    - Cyg A gives k = 1.00, with R = 0.97.
    - A burst of S/N 40 at DM 50, 0.30° from its pixel's centre, is measured at 0.94 ± 0.05 of its fluence. Its S/N is 34.7.
    - A burst of S/N 120 is measured at 1.03 ± 0.02, with S/N 104 and a fitted position 0.03° from the truth.
    - Over 32 bursts at pixel centres, edges, vertices and random positions, the measured fluence is 1.013 ± 0.009 of the truth, against 0.873 ± 0.012 before the corrections. Sixteen bursts of S/N 120 give 1.008 ± 0.005. The remaining systematic error is estimated at about 1%.
    - Before these corrections, the test burst was measured at 0.82 of its fluence. The tier high-pass took 9%, the pixel offset and interpolation 5%, and the noise 6%. Two calibrator errors, −6.8% for the pixel and +5.3% for the background, made k = 0.98 by chance.

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

Monitors keep the data of RFI candidates too, unless the RFI is certain (DM < 2, or much of the sky at once). Each candidate also carries in-band quality measures (`quality.py`). Only class `candidate` (and coordinated injections) opens events; candidates at known sources (`known`) do not. On the node, the archiver (`archiver.py`) removes `known` candidates that are not archived after `--review-days` (14). The archive on the shared file system is curated by `archive_curator.py` (job `epic-archive-curator`): archived candidates and events, like the imager's long integrations, are kept for five years and never removed while their event awaits review (DESIGN.md §5.3).

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
python -m skymon combine --nodes epic1=http://epic1:8095,...,epic6=http://epic6:8095 --db-port 5433 --out combiner --port 8096 --inject-every 6
python -m skymon frblist --out frbs.json --start $(date +%s) --duration 3600   # the same FRBs for every band's simulator
python -m skymon simtest --realtime --frb-file frbs.json --chan0 1378 ...     # one per band, on its own worker
```

`POST` requests need the header `X-Skymon-Token` when `SKYMON_TOKEN` is set.

## Live test injections

Every `--inject-every` minutes on average (Poisson), the monitor adds a simulated dispersed burst to the incoming image, before regridding.
- **Where.** A random tracked position that stays up for the whole sweep.
- **DM.** Log-uniform over what that position can still complete.
- **Strength.** A target S/N of 10–30, set from the measured noise of that pixel, for an ideal search of the burst as injected.
- **Scattering.** Scattered by default (`--inject-scatter`): the scattering time at 600 MHz is log-uniform over 0.31–10 ms (`--inject-tau600-ms`) and scales as ν⁻⁴, a one-sided exponential in each channel. A fraction 0.2 (`--inject-unscattered`) is left unscattered as a check of short pulses. `tau_1ghz` and `tau_s` (at the band centre) are logged; the recovered over the target S/N is the search efficiency for that burst.

A candidate that matches an injection is classified as `injection`. Every injection is logged twice in `<out>/injections.jsonl`: when it is scheduled, and again as recovered or missed. The detection efficiency of the running system is therefore measured continuously; `/api/status` reports the totals.

## Use

The `epic310` environment has everything the monitor needs: numpy, astropy, healpy, lsl and gcc. The compiled kernels (`_kernels.c`) are built on first use into the user's cache directory (`$XDG_CACHE_HOME/skymon/<host>/`).

```bash
export PYTHONPATH=/path/to/LWA_EPIC/src/python
python -m skymon run --shm epic_sky0 --out /data/skymon --port 8095 --threads 8
python -m skymon simtest --duration 1800 --realtime --out simtest     # simulated sky, then evaluation
python -m skymon simulate --shm skysim --duration 3600                    # images only, at the real rate
python -m skymon evaluate --truth simtest/truth.jsonl --out simtest
```

With 8 threads on 12 cores the monitor takes 0.57 of real time at 41.8 ms images with `--max-delay-factor 2 --block 512` (0.32 at 83.6 ms) on a Threadripper PRO 5965WX; 12 threads take 0.52, and more threads add nothing, since they then compete for memory bandwidth. Confined to 8 cores and their siblings (the production layout), it takes 0.66 before and 0.67 with the wide-pulse search; confined to one L3 group of 6 cores it takes 0.87 (docs/skymon/DESIGN.md, §10.1c). The default OpenMP wait policy is faster than `OMP_WAIT_POLICY=passive` by about 15% (docs/skymon/DESIGN.md, §10.1b).

`epic++` needs `--sky_shm epic_sky0` to publish. The ring holds 64 images by default (5.3 s at 83.6 ms, 537 MB for 128 channels of 128 × 128 pixels; 2.7 s and 3.2 GB for 192 channels of 256 × 256 pixels).

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
cd tests/python && python -m unittest -v test_known test_chanflag test_skymon test_combiner test_coincidence test_context test_conditions test_localdb test_flux test_archive_curator test_save_output
```

The tests cover:
- the FDMT against numpy and brute force;
- the FDMT for any number of channels: the plan of 192 channels, the kernel against numpy for 1 to 192 channels, the search against a reference, a pulse keeping the same fraction of its fluence in 192 channels as in 128, the S/N rising as sqrt(192/128), and the real band edges;
- tier histories in half precision (search, cascade and shift against single precision; a burst found with its S/N within 0.2%), and the vector paths of the regridding and the normalisation against references;
- tier coverage;
- the shared-memory ring, including overruns and writer restarts;
- the C++ publisher's pixel order against `epic_utils.save_output`;
- the geometry;
- detection by the searcher;
- the wide-pulse search: the trend fitted over a longer history than the transform, the high-pass per tier, the tier settings and the false-alarm threshold, the gain on a scattered burst without loss on a short one, rising pixels with the wide boxcars, and the noise at fixed widths; the exponential scattering model and the scattered live injections;
- interference flagging (`test_chanflag.py`): one channel and one image, noise without flags, the persistent mask and its release, and a moving carrier in a simulated sky that leaves no candidate while a burst is found;
- an end-to-end simulated sky with stored candidates and the HTTP interface, in 64 channels and in 192;
- live injections;
- known-source labels (Sun, pulsars, bright sources) and the narrowband classification (`test_known`);
- a monitor that waits for a missing imager and survives a restart and a gap;
- the combiner's consensus, and the external context (parsing of saved SWPC samples in `tests/python/data/`, matching, lightning, unreachable sources);
- conditions during a minute of threefold noise, over HTTP, in the combiner's events, and in the database through an outage;
- flux (`test_flux`): the beam tables, the fluence arithmetic and its error, the calibration records, the peak fit, the exact undoing of the tiers' high-pass, and a simulated sky in which Cyg A gives k = 1, a burst of S/N 40 has its fluence within 3 errors and 3%, and a burst of S/N 120 is found at its position with its fluence despite the clip.
