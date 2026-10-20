# What the all-sky monitor needs from the streaming calibrator

The all-sky monitor (skymon) gives every candidate a fluence and a mean flux density. The images it searches carry no absolute flux scale. The gain tables of the streaming calibrator are relative: in each channel and polarisation their median amplitude is one, so that calibration changes neither the image bandpass nor the numeric range of the half-precision gridding. The imager applies no primary beam. This document states what the monitor requires from the calibrator, why, and in what form. It lists the questions that remain open; it does not fix the calibrator's design.

## 1. How the monitor uses the calibration

With a gain table applied, Stokes I (XX + YY) of an unpolarised source of flux density $S$ in direction $s$ appears in the image as

$$
I(s) = k\,S\,B_\mathrm{eff}(s), \qquad B_\mathrm{eff}(s) = \tfrac{1}{2}\left(F_X^2\,B_X(s) + F_Y^2\,B_Y(s)\right),
$$

where:

- $F_X$, $F_Y$ is the absolute scale that the relative table removed, per channel;
- $B_X$, $B_Y$ is the power response of the array towards $s$;
- $k$ collects everything internal to the imager: gridding, transform, accumulation over the image and quantisation.

The monitor measures $k$ itself, every minute and in every channel, from the calibrators in its own images: the excess of a calibrator's pixel over its surroundings, divided by $S\,B_\mathrm{eff}$ towards the calibrator. A candidate's fluence then follows from its signal, the noise of its pixel and $k\,B_\mathrm{eff}$ towards the candidate (`src/python/skymon/flux.py`).

As a consequence, only ratios of the beam enter, between a calibrator's direction and the candidate's direction. The normalisation of the beam is free, and no constant of the imager is needed. What the monitor cannot measure by itself is:

- the absolute scale $F$, which the calibrator removes from the tables;
- the true flux of the calibrators;
- in time, the beam as calibrated by direction-dependent calibration.

## 2. Requirements

| Id | Quantity | Why | Needed from |
|---|---|---|---|
| C1 | `gain_version` of each table, and of each image | A candidate is converted with the calibration its images were made with. The imager already records the version in the image metadata (`gain_version`), and the sky ring carries it to the monitor. | first use |
| C2 | Absolute scale $F_X$, $F_Y$ per channel, with its units (data counts per Jy$^{1/2}$ as solved) | The tables are normalised per channel; without $F$ the polarisations, and channels with different scales, cannot be combined. | first use |
| C3 | The calibrators of each solution: name, J2000 position, the model spectrum used (per channel, or the parameters of the spectral model), and the flux scale solved for each | The monitor measures $k$ from them in the images. A source whose flux the calibrator solved (the Sun, or a scintillating or flaring source) must be marked as such, so that it is used with its solved flux or not at all. | first use |
| C4 | Uncertainty of C2 and C3 (per channel, or a single figure) | The fluence error includes the noise of the candidate's contributions (`flux.py`) but no calibration uncertainty. | when available |
| C5 | Quality of each solution: χ²/dof per channel and the cut applied, channels rejected as interference, stands flagged | A candidate whose channels were rejected by the calibrator, or that falls in a poor solution, is marked for review; the monitor's own channel flags can be compared. | when available |
| C6 | The primary beam: power response for X and Y on a grid of elevation and azimuth, at three or more frequencies across the band | Only the beam's shape matters (section 1). Until it is calibrated the monitor uses the lsl dipole model ("empirical"), computed by itself. | after direction-dependent calibration |
| C7 | Direction-dependent effects that change the response of the images to a point source in a given direction (for example, decorrelation by residual direction-dependent phases), as a multiplicative factor on the same grid as C6 | A calibrator and a candidate in different directions see different responses. If direction-dependent calibration is applied by prediction and subtraction in the image domain, the response to a new burst is unchanged and this factor is one. | after direction-dependent calibration |

### Beam table (C6, C7)

The monitor reads a beam table in this form; any grid is accepted, provided that the elevations ascend and the azimuths cover 360° in equal steps:

| Field | Content |
|---|---|
| `source` | `model` or `calibrated` |
| `model` | name of the model or of the calibration that produced it |
| `version` | increases with every change of the table |
| `freqs_mhz` | frequencies of the planes (three or more across the band) |
| `alt_deg`, `az_deg` | elevation (ascending) and azimuth (from north through east, uniform steps) of the grid |
| `X`, `Y` | power response `[freq][alt][az]`; the normalisation is free (section 1) |

A table that includes C7 states this in `model`. The monitor reports the `source`, `model` and `version` of the beam with every fluence. A beam table delivered with the calibration takes precedence over the monitor's own model.

## 3. Delivery

- **One record per gain table, keyed by `gain_version`.** Versions must increase monotonically, also across restarts of the calibrator, since the monitor keeps a history of records by version.
- **Delivery on the same node as the imager and the monitor.** A small file in shared memory beside the gain table, replaced atomically (written to a temporary name and renamed), in JSON. The monitor checks the file for a change every 10 s of data (`--calinfo`), so of records replaced faster than that only the last is read; it accepts a record without the optional fields (C4–C7), and a missing or unreadable record leaves candidates without a flux, never stopping the search.
- **The beam table (C6, C7)** changes far less often than the gains. It may be a separate file with its own version, referenced from the record by version, rather than repeated in every record.
- **Persistence.** With a local database (`--db`), the monitor stores every record it reads, so the calibration of every candidate can be reconstructed later. The calibrator need not keep a history for the monitor.
- **Cadence.** Every change of `gain_version` in the images makes the monitor hold 24 images and rebase every pixel's baseline before searching them ([DESIGN.md §6.1](DESIGN.md#61-gain-version-in-the-images)). The number of published tables is therefore a cost to the search; the calibrator publishes a table only after a significant change or every 600 s.
- **Independence.** The calibrator must never wait for the monitor, and the monitor never for the calibrator.

The streamcal daemon writes C1–C3 and C5 with every published gain table when started with `--calinfo` (`src/python/streamcal/calinfo.py`; [streaming calibration, section 4.4.3](../streaming_calibration/04_implementation.md#443-the-calibration-record)). In the daemon's end-to-end test the absolute scale it publishes equals the true median gain amplitude to 0.8–1.0%. It writes no uncertainty (C4) and no beam table (C6, C7), so the monitor uses its own dipole model. The record also carries each calibrator's elevation, azimuth and solved position offset (`alt_deg`, `az_deg`, `offset_deg`), the solved channels, and the names of the model and solver.

## 4. Open questions

1. Is $F$ (C2) well defined per channel after direction-dependent calibration? With direction-dependent screens, is the relative table still a single correction per stand and channel, and does a single absolute scale per channel remain meaningful?
2. Will direction-dependent effects be corrected in the gains applied by the imager, or by prediction and subtraction in the image domain? The answer decides whether C7 is needed.
3. The calibrator solves the A-team (Cas A, Cyg A and Vir A with the de Gasperin et al. 2020 spectra, Tau A with its earlier model) and the Sun as a free flux, and reports the solved scale of each. Should the monitor use the solved scales of all calibrators, or only those with catalogue spectra?
4. How is the calibrated beam (C6) to be parameterised: on an elevation and azimuth grid as above, as coefficients of a model, or per stand?
5. The record path is chosen per daemon (`--calinfo`), by convention `/dev/shm/epic_calinfo_gpu<N>.json` beside the gain table of GPU N. With several imaging pipelines on one node, each monitor must be given the record of its pipeline.
6. What uncertainty (C4) can the calibrator provide, and at what cadence?
