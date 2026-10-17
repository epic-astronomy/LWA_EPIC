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
| C4 | Uncertainty of C2 and C3 (per channel, or a single figure) | The fluence error at present includes only the S/N. | when available |
| C5 | Quality of each solution: χ²/dof per channel and the cut applied, channels rejected as interference, stands flagged | A candidate whose channels were rejected by the calibrator, or that falls in a poor solution, is marked for review; the monitor's own channel flags can be compared. | when available |
| C6 | The primary beam: power response for X and Y on a grid of elevation and azimuth, at three or more frequencies across the band | Only the beam's shape matters (section 1). Until it is calibrated the monitor uses the lsl dipole model ("empirical"), computed by itself. | after direction-dependent calibration |
| C7 | Direction-dependent effects that change the response of the images to a point source in a given direction (for example, decorrelation by residual direction-dependent phases), as a multiplicative factor on the same grid as C6 | A calibrator and a candidate in different directions see different responses. If direction-dependent calibration is applied by prediction and subtraction in the image domain, the response to a new burst is unchanged and this factor is one. | after direction-dependent calibration |

### Beam table (C6, C7)

The monitor reads a beam table in this form; any grid is accepted, provided that the azimuth steps are uniform:

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
- **Delivery on the same node as the imager and the monitor.** A small file in shared memory beside the gain table, replaced atomically (written to a temporary name and renamed), in JSON. The monitor reads the file when it changes (`--calinfo`); it accepts a record without the optional fields (C4–C7), and a missing or unreadable record leaves candidates without a flux, never stopping the search.
- **The beam table (C6, C7)** changes far less often than the gains. It may be a separate file with its own version, referenced from the record by version, rather than repeated in every record.
- **Persistence.** The monitor stores every record it receives in its local database, so the calibration of every candidate can be reconstructed later. The calibrator need not keep a history for the monitor.
- **Independence.** The calibrator must never wait for the monitor, and the monitor never for the calibrator.

A prototype writer for C1–C3, C5 and a model beam table was tested against the calibrator's synthetic end-to-end test: the absolute scale it published agreed with the true gain amplitudes to within 1.1% in both polarisations. It is not part of either branch, and this document, not the prototype, defines what the monitor needs.

## 4. Open questions

1. Is $F$ (C2) well defined per channel after direction-dependent calibration? With direction-dependent screens, is the relative table still a single correction per stand and channel, and does a single absolute scale per channel remain meaningful?
2. Will direction-dependent effects be corrected in the gains applied by the imager, or by prediction and subtraction in the image domain? The answer decides whether C7 is needed.
3. Which calibrators will the calibrator solve for, and with which model spectra (the present A-team models, or the de Gasperin et al. spectra)? Is Cas A's flux, found about 1.5 times the Baars model, to be taken from the solved scale or from an updated model?
4. How is the calibrated beam (C6) to be parameterised: on an elevation and azimuth grid as above, as coefficients of a model, or per stand?
5. With several imaging pipelines on one node, how is the record of each pipeline named, so that each monitor reads its own?
6. What uncertainty (C4) can the calibrator provide, and at what cadence?
