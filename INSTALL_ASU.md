# General

Start with the general [Enterprise new user setup](https://docs.google.com/document/d/1DOoiYEZd15KMw4KNSIkaGnph2aARI4n77cEuQHiOJqs/edit), and do the steps up to and including anaconda installation. When you create your environment, **be sure to use python 2**, not 3 as is the default. Give an appropriate name to the conda environment. Below we use `epic` as the example.

After the installation of anaconda do the following
```
$ user=`whoami`
$ env=epic
 $ conda create -n ${env} python=2.7
$ conda activate ${env}
$ conda install ipython
$ python -m ipykernel install --user --name ${env} --display-name ${env}
$ pip install lsl
```

# Bifrost
Until some pull requests are merged, we need to use a hybrid of two bifrost forks.
```
$ cd ~/src
$ git clone https://github.com/ledatelescope/bifrost.git -b plugin-wrapper bifrost
$ git clone https://github.com/KentJames/bifrost.git -b optim_romein bifrost_jkent
$ cd bifrost  # work inside the LEDA bifrost
$ cp ~/src/bifrost_jkent/src/romein.cu src/
$ cp /data4/jdowell/CodeSafe/bifrost/src/proclog.cpp src/  # Make multi-user installation work
```
(The `proclog.ccp` file referenced above is also in this repository, `LWA_EPIC/config/ASU_proclog.cpp`, so you can copy from there in case anything ever happens to the one on `/data4/`.)

Check `README.md` to install dependencies (do this in your epic env).
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
$ cp ~/src/LWA_EPIC/config/ASU_user.mk ~/bifrost/user.mk
```

Finally, install (note **this must be done on intrepid with your environment activated**):
```
$ make -j 32
$ make install INSTALL_LIB_DIR="/home/${user}/src/anaconda/envs/${env}/lib" INSTALL_INC_DIR="/home/${user}/src/anaconda/envs/${env}/include" PYINSTALLFLAGS="--prefix=/home/${user}/src/anaconda/envs/${env}"
```

### Testing multiple branches
If you want to install and test multiple bifrost branches, it is encouraged to use different conda environments. Check the environment variables of the install directory to properly link with the libraries.

# EPIC
https://github.com/epic-astronomy/epic
