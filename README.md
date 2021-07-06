![](./image_compare.gif)

# What is EPIC?


A direct imaging correlator for radio interferometer arrays. Instead of cross-correlating voltages streams from all antenna to form visibilities, voltages can be gridded and Fourier Transformed directly into the image domain. A Discrete Fourier Transform (DFT) can also be implemented to skip the gridding step.
However, as seen in the gif above, the FFT and DFT implementations currently have different normalizations.

The process creates all sky images in real time with millisecond time resolution. High time resolution is vital in the identification and classification of radio transients, FRBs, Pulsar timing, and Gravitational Wave follow-ups.

Images can easily be combined to form deeper integrations as well when lower sensitivity is desired for weaker signals.

This repository is an implementation of the EPIC (E-field Parallel Imaging Correlator) and EPIC-DFT correlator specifically for use with the Long Wavelength Array (LWA).

For a generalized implementation of EPIC please see [our EPIC repo](https://github.com/epic-astronomy/EPIC).


# Installation
Testing and code development can be accomplished offline using the `intrepid` machine on the ASU LoCo cluster. Specific install instructions can be found [here](INSTALL_ASU.md)
