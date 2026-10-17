# 9. Primary-beam correction

## 9.1 What EPIC images contain

An EPIC image is the sky multiplied by the dipole power pattern of each polarisation:

```math
I'_{XX}(\mathbf l) \propto B_X(\mathbf l, f)\, I(\mathbf l, f),\qquad I'_{YY}(\mathbf l) \propto B_Y(\mathbf l, f)\, I(\mathbf l, f),
```

convolved with the synthesised beam. epic++ divides out only the gridding kernel's response (`gcf_correction_grid`, section 1.1). Neither epic++ nor its Python helpers apply a primary-beam correction: no code in `src/` refers to the dipole beam. Images are in instrumental units (`BUNITS = UNCALIB`).

## 9.2 How large the beam effect is for the watched sources

From the LSL `empirical` dipole model, normalised to 1 at zenith, for 2026-10-17 at LWA-SV and only while the source is above 15° elevation:

| Source | Elevation range | Stokes I factor, 38 MHz | Change over the day, 38 MHz | XX/YY, 38 MHz | Change over the day, 74 MHz |
|---|---|---|---|---|---|
| Crab | 16–78° | 0.13–0.96 | ×7.5 | 0.97–1.64 | ×5.8 |
| PSR B0809+74 | 19–50° | 0.16–0.62 | ×3.9 | 0.57–0.76 | ×3.1 |
| PSR B1133+16 | 17–71° | 0.14–0.90 | ×6.6 | 0.94–1.71 | ×5.1 |
| Cyg A | 16–83° | 0.13–0.99 | ×7.8 | 1.00–1.25 | ×6.2 |
| Sun | 15–46° | 0.12–0.56 | ×4.8 | 0.73–1.41 | ×3.7 |

A watched source's apparent brightness therefore changes by a factor of 3–8 over a day purely through the beam. The X and Y dipoles see it differently by up to 1.7×, and the ratio depends on azimuth.

## 9.3 Is it required, and will it help?

**Detection: no.** Dividing a pixel by $`B(\mathbf l)`$ scales its signal and its noise equally. The S/N of a transient or a pulse is unchanged, so detection thresholds should stay in S/N units, which the FRB and pulsar searches use.

**Flux densities, fluences and light curves: yes.** Without correction:

- **Light curves:** a pulsar or the Crab followed through a day varies by ×3–8 from the beam alone. Long-term pulsar monitoring (section 2 of `pulsars_and_calibration.md`) needs the correction to compare epochs.
- **Fluence limits and fluences:** these are biased low by $`B`$, and the bias depends on elevation. The FRB fluence limits in the notes assume zenith.
- **Spectra:** the beam's frequency dependence across 16–88 MHz enters measured spectral indices.
- **Stokes I:** forming I as XX+YY weights X and Y unequally where $`B_X\ne B_Y`$, up to 1.7× here, and mixes polarised emission into I.

**Image quality: no.** A primary-beam correction does not change calibration, sidelobes or artefacts. The calibration already includes the beam in its sky model (sections 2 and 6).

## 9.4 How to apply it

The correction belongs downstream of the GPU, applied per image, channel and polarisation:

```math
I_{\rm Jy}(\mathbf l) = \frac{1}{2}\left[\frac{I'_{XX}(\mathbf l)}{s_X\,B_X(\mathbf l)} + \frac{I'_{YY}(\mathbf l)}{s_Y\,B_Y(\mathbf l)}\right],
\qquad B > B_{\min}\ (\text{elevation} \gtrsim 15^\circ),
```

where $`s_{X,Y}`$ are the absolute flux scales the calibration service solves (counts per Jy, kept by `daemon.table()`, section 4.3).

Where it fits in the pipeline:

- **Pixel extractor:** multiply the extracted values of a watched source by $`1/(s\,B)`$ at its exact position (`source_pix`).
- **FITS writer:** store per-channel beam maps, or a beam-corrected Stokes I plane, with a header keyword for the flux scale.

The cost is negligible. The beam changes only with frequency in the station frame, so the maps are fixed per channel.

Below about 15° elevation $`B`$ falls to 0.1 and the correction amplifies noise and model errors, so pixels there should be masked or carry the beam as a weight.

## 9.5 Accuracy is limited by the beam model

The real data show the limit. With the LSL beam in the model, the fitted Cyg A flux (section 8.5) moved from 0.6–0.8 of the catalogue near zenith (04:00 UTC) to 1.1–1.3 two hours later (06:00 UTC). Part of that is the catalogue spectrum, but the change with elevation points to beam-model errors of tens of per cent.

The calibration service already solves an apparent flux scale for every bright source in every 10 s table (section 6.2). Those scales are samples of $`B_{\rm true}/B_{\rm model}`$ along the A-team tracks, about 360 per hour per source per channel. Accumulated over days, they give an empirical correction to the beam model in azimuth and elevation, the method used to bootstrap LWA and OVRO-LWA beams from A-team transits. The A-team catalogue spectra (de Gasperin et al. 2020) must be fixed first, or the catalogue error enters the beam.

## 9.6 Recommendation

1. Add the per-channel X and Y beam maps and the flux scales to the image metadata.
2. Apply the correction in the pixel extractor for watched sources, so that light curves and fluences are in Jy.
3. Keep the search statistics in S/N units.
4. Log the per-source flux scales from streamcal, and refine the beam model from them once the A-team spectra are updated.
