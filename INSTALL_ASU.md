# General

Start with the general [Enterprise new user setup](https://docs.google.com/document/d/1DOoiYEZd15KMw4KNSIkaGnph2aARI4n77cEuQHiOJqs/edit), and do the steps up to and including anaconda installation. When you create your environment, **be sure to use python 3**. Give an appropriate name to the conda environment. Below we use `epic` as the example.


After the installation of anaconda do the following
```
$ user=`whoami`
$ env=epic
 $ conda create -n ${env}
```
It is advised to install the dependencies of LSL before attempting a `pip install`. While pip generally is good about grabbing package dependencies, the configuration of some packages requires dependencies to be installed before pip can check if the dependencies are installed.

While not explicitly mentioned, lsl depends on `astropy` for fits I/O. Inclusion of astropy is not a mistake.

To install dependencies for lsl excute the following lines. This may not be necessay depending on your environment but will help smooth the install.
```
$ conda activate ${env}
$ conda install -c free atlas
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
```

Note that the `README.md` in bifrost can be (and as of this commit *is*) out of date. Use the following line to install dependencies. (Do this in your epic env, *not* with sudo as the bifrost README indicates). The ctypes gen commit referenced here works for Ubuntu >18. In the future this may be another point of trouble. If this version doesn't work, you will want to file and issue on the [ledatelescope bifrost repo](https://github.com/ledatelescope/bifrost/issues).
```
$ pip install contextlib2 pint git+https://github.com/olsonse/ctypesgen.git@9bd2d249aa4011c6383a10890ec6f203d7b7990f
```

Now we need to actually install bifrost. First set up configuration for intrepid. Copy the file `LWA_EPIC/config/ASU_user.mk` from this repository to your bifrost directory:
```
$ cp ~/src/LWA_EPIC/config/ASU_user.mk ~/src/bifrost/user.mk
```

Before proceeding to compiling, we need to ensure your `PATH` is setup to find the cuda libraries. Add the following lines to your `.bashrc` (or relevant shell initialization script):
```
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
```
Then source the file and reactivate your environment:
```
$ source ~/.bashrc
$ conda activate ${env}
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
$ cd ~/src/LWA_EPIC/LWA
$ python LWA_bifrost.py --offline --tbnfile=/data5/LWA_SV_data/data_raw/TBN/Jupiter/058161_000086727 --imagesize 64 --imageres 1.79057 --nts 512 --channels 4 --accumulate 50 --ints_per_file 40
```
This will generate a series of `.npz` files with example images. Have a look at them
to be sure they look like the sky!
