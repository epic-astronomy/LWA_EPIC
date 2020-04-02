# General

Start with the general [Enterprise new user setup](https://docs.google.com/document/d/1DOoiYEZd15KMw4KNSIkaGnph2aARI4n77cEuQHiOJqs/edit), and do the steps up to and including anaconda installation. When you create your environment, **be sure to use python 2**, not 3 as is the default. Give an appropriate name to the conda environment. Below we use `epic` as the example.


After the installation of anaconda do the following
```
$ user=`whoami`
$ env=epic
 $ conda create -n ${env} python=2.7
```
It is advised to install the dependencies of LSL before attempting a `pip install`. While pip generally is good about grabbing package dependencies, the configuration of some packages requires dependencies to be installed before pip can check if the dependencies are installed.

While not explicitly mentioned, lsl depends on `astropy` for fits I/O. Inclusion of astropy is not a mistake.

To install dependencies for lsl excute the following lines. This may not be necessay depending on your environment but will help smooth the install.
```
$ conda activate ${env}
$ conda config --add channels conda-forge
$ conda install aipy scipy numpy fftw astropy healpy ephem pytz matplotlib
$ pip install lsl
```

# Bifrost
```
$ cd ~/src
$ git clone https://github.com/epic-astronomy/bifrost.git
$ cd bifrost 
# For development work, you will want to checkout the appropriate branch. For example:
$ git checkout plugin-wrapper
$ cp /data4/jdowell/CodeSafe/bifrost/src/proclog.cpp src/  # Make multi-user installation work
```
(The `proclog.ccp` file referenced above is also in this repository, `LWA_EPIC/config/ASU_proclog.cpp`, so you can copy from there in case anything ever happens to the one on `/data4/`.)

Check `README.md` to install dependencies (do this in your epic env, *not* with sudo as the README indicates).
Always check that the ctypesgen is the latest as mentioned in the README.md
```
$ pip install numpy contextlib2 pint git+https://github.com/olsonse/ctypesgen.git@9bd2d249aa4011c6383a10890ec6f203d7b7990f
```

Downgrade matplotlib
```
$ conda install matplotlib=2.2.3
```

Now we need to actually install bifrost. First set up configuration for intrepid. Copy the file `LWA_EPIC/config/ASU_user.mk` from this repository to your bifrost directory:
```
$ cp ~/src/LWA_EPIC/config/ASU_user.mk ~/src/bifrost/user.mk
```

Finally, install (note **this must be done on intrepid with your environment activated**):
```
$ make -j 32
$ make install INSTALL_LIB_DIR="/home/${user}/src/anaconda/envs/${env}/lib" INSTALL_INC_DIR="/home/${user}/src/anaconda/envs/${env}/include" PYINSTALLFLAGS="--prefix=/home/${user}/src/anaconda/envs/${env}"
```

### Testing multiple branches
If you want to install and test multiple bifrost branches, it is encouraged to use different conda environments. Check the environment variables of the install directory to properly link with the libraries.

# Test Run
Remember to check that you are logged onto intrepid and in your environment.
```
$ cd ~/src/LWA_EPIC/scripts
$ python LWA_bifrost.py --offline --tbnfile=/data5/LWA_SV_data/data_raw/TBN/Jupiter/058161_000086727 --imagesize 64 --imageres 1.79057 --nts 512 --channels 4 --accumulate 50 --ints_per_file 40
```
This will generate a series of `.npz` files with example images. Have a look at them
to be sure they look like the sky!
