# What is EPIC?
![](https://github.com/epic-astronomy/LWA_EPIC/raw/main/image_compare.gif)

A direct imaging correlator for radio interferometer arrays. Instead of cross-correlating voltages streams from all antenna to form visibilities, voltages can be gridded and Fourier Transformed directly into the image domain. A Discrete Fourier Transform (DFT) can also be implemented to skip the gridding step.
However, as seen in the gif above, the FFT and DFT implementations currently have different normalizations.

The process creates all sky images in real time with millisecond time resolution. High time resolution is vital in the identification and classification of radio transients, FRBs, Pulsar timing, and Gravitational Wave follow-ups.

Images can easily be combined to form deeper integrations as well when lower sensitivity is desired for weaker signals.

This repository is an implementation of the EPIC (E-field Parallel Imaging Correlator) specifically for use with the Long Wavelength Array (LWA). In this new version of the code, the GPU implemetation is optimized to produce images completely using on-chip memory for all polarizations. See [EPIC Memo #9](https://github.com/epic-astronomy/Memos/blob/temp/dx_optimizations/PDFs/009_EPIC_Code_Optimizations.md) for details on the new optimizations.

For a generalized implementation of EPIC please see [our EPIC repo](https://github.com/epic-astronomy/EPIC).


# Build Instructions
Download the source code with all the dependencies. 
```bash
git clone -b test/4090 --recurse-submodules -j5 https://github.com/epic-astronomy/LWA_EPIC.git
```

It is recommended to build and run EPIC in a separate conda environment. Use the `conda_env.yml` file to create a new conda enviroment and install all the necessary packages. Change the last line in this file to the desired install location and exceute the following command.
```bash
conda create -n epic python=3.10 --file=conda_env.yml
```
Additionally, download and extract [NVIDIA MathDx](https://developer.nvidia.com/mathdx#:~:text=wget%20https%3A//developer.download.nvidia.com/compute/mathdx/redist/mathdx/linux%2Dx86_64/nvidia%2Dmathdx%2D22.11.0%2DLinux.tar.gz) package to the `src/extern` folder. 

Within this environment, 

Run the following shell commands to build EPIC
```bash
cd LWA_EPIC
mkdir build && cd build
cmake ..
make
```

To create a debug build:
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

Memory errors can be detected using the following valgrind command
```bash
valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes  \
         --verbose \
         --log-file=valgrind-out.txt \
          ./epic++
```

This creates an executable called `epic++`. If it needs to executed from a different directory please also copy all the python files from the build directory or ensure they are visible to the embedded python interpreter.
