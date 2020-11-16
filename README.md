This is an implementation of the EPIC (E-field Parallel Imaging Correlator) and EPIC-DFT correlator specifically for use with the Long Wavelength Array (LWA).

For a generalized implementation of EPIC please see [our EPIC repo](https://github.com/epic-astronomy/EPIC).

# What is EPIC?
A direct imaging correlator for radio interferometer arrays. Instead of cross-correlating voltages stream from all antenna stream to form visibilities, voltages can be gridded and Fourier Transformed directly into the image domain. A Discrete Fourier Transform (DFT) can also be implemented to skipping the gridding step.

The process creates images in real time with milisecond time resolution and deeper integrations can also be performed. High time resolution is vital in the identification and classification of radio transients, FRBs, Pulsar timing, and Gravitational Wave follow-ups.

# Installation
Testing and code development can be accomplished offline using the `intrepid` machine on the ASU LoCo cluster. Specific install instructions can be found [here](INSTALL_ASU.md)
