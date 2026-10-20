# Deployment as a container

The imager (`epic++`), the streaming calibration daemon (`streamcal`) and the
all-sky monitor (`skymon`) are distributed as one container image,
`ghcr.io/epic-astronomy/lwa-epic`. GitHub Actions builds it from the
[Dockerfile](../Dockerfile) on every push to `phoenix` and `phoenix-fx`
([images workflow](../.github/workflows/images.yml)). Each commit gets the
tag `sha-<short sha>` and each branch a tag of its name. `latest` follows
`phoenix`.

## Contents of the image

| Path | Contents |
|---|---|
| `/opt/epic/epic++` | the imager, with file capabilities `cap_net_raw,cap_ipc_lock+ep` |
| `/opt/epic/*.py`, `/opt/epic/epic_grpc` | modules of the embedded Python interpreter |
| `/opt/epic/streamcal`, `/opt/epic/skymon` | calibration daemon and all-sky monitor (`python3 -m ...`) |
| `/usr/local/lib/epic` | shared libraries built with `epic++` (glog, prometheus-cpp, pqxx, Tensor-Core Correlator) |
| `/usr/local/cuda/include` | CUDA headers, needed by NVRTC when the Tensor-Core Correlator compiles its kernel at start-up |

The image is built for CUDA 12.3, which needs an NVIDIA driver of release 545
or later on the host, and for compute capability 8.9 (`CUDA_ARCH`). The build
arguments `EPIC_NCHAN_SELECT` (default 192) and `EPIC_VERBS_NPKTBUF`
(default 32768) set the corresponding CMake options. The Python packages are
pinned in [docker/requirements.txt](../docker/requirements.txt), installed
from binary wheels, and
[docker/requirements-source.txt](../docker/requirements-source.txt), the
packages without wheels (LSL, AIPY, PyGDSM), built from source against them.
A directory `docker/wheelhouse` (not in the repository), when present, holds
wheels downloaded beforehand, which the build uses in place of downloads.

The Tensor-Core Correlator compiles its kernel with NVRTC when the imager
starts, so the image carries the CUDA headers and a link
`libnvrtc-builtins.so` (the unversioned name cudawrappers loads, otherwise
provided only by the toolkit's development package) in a default library
directory.

## Privileges

The verbs packet receiver opens a raw-packet queue, which requires
`CAP_NET_RAW`, and locks its receive buffers in memory, which uses
`CAP_IPC_LOCK` when the memlock limit is finite. On the file-based
deployment these were granted by running `setcap` on the extracted binary
at every start. In the image, the capabilities are attached to `/opt/epic/epic++` when the image is built
(`setcap cap_net_raw,cap_ipc_lock+ep`), and extended attributes are kept in
the image layers. Nothing is set on the host at deployment.

File capabilities take effect only within the container's capability bounding
set. The container therefore runs as the unprivileged user `epic`
(uid 1000) and is given exactly these two capabilities in addition to the
runtime's defaults (`--cap-add NET_RAW --cap-add IPC_LOCK`, or `cap_add` in
Nomad). It does not run privileged. Because `epic++` is then a
secure-execution binary, the dynamic loader ignores `LD_LIBRARY_PATH`, so the
image registers its library directory with `ldconfig`.

## Requirements on the host

| Requirement | Docker option | Reason |
|---|---|---|
| GPU | `--gpus all` or `--runtime nvidia` | imaging |
| host network | `--network host` | multicast reception, metrics, video output |
| RDMA devices | `--device /dev/infiniband/uverbs0 --device /dev/infiniband/uverbs1 --device /dev/infiniband/rdma_cm` | verbs receiver |
| capabilities | `--cap-add NET_RAW --cap-add IPC_LOCK` | verbs receiver |
| memlock | `--ulimit memlock=-1` | pinned receive and transfer buffers |
| host IPC namespace | `--ipc host` | shared-memory rings in `/dev/shm` between the imager, the calibration daemon and the all-sky monitor, which run in separate containers |
| output directories | `-v <dir>:<dir>` | FITS integrations, caches, sky models |

Under Nomad, the Docker plugin of each imaging node must allow the two
capabilities, bind mounts and the `nvidia` runtime. Docker must know the
NVIDIA runtime. The devops repository prepares both (playbook
`34_Epic_container_hosts.yml`) and runs the imager and the calibration daemon
as the job `epic-imager` (`epic-imager-container.nomad.j2`, playbook
`35_Epic_imager_container_deployment.yml`).

## Running by hand

A synthetic-data benchmark, which needs only the GPU:

```bash
docker run --rm --gpus all ghcr.io/epic-astronomy/lwa-epic:phoenix-fx \
  ./epic++ --bench --bench_ngulps 60 --imager fx --imagesize 256 --imageres 0.528 \
  --nts 1000 --seq_accum 41 --channels 192 --support 5 --aeff 25 --chan_nbin 1 \
  --nstreams 8 --kernel_oversample 4 --wproj --remove_autocorr --ngpus 1 --gpu_ids 0 \
  --disable_metrics
```

The imager on a live stream:

```bash
docker run --rm --gpus all --network host --ipc host \
  --device /dev/infiniband/uverbs0 --device /dev/infiniband/uverbs1 --device /dev/infiniband/rdma_cm \
  --cap-add NET_RAW --cap-add IPC_LOCK --ulimit memlock=-1 \
  -v /data/accumulations:/data/accumulations \
  ghcr.io/epic-astronomy/lwa-epic:phoenix-fx ./epic++ <options> --out_dir /data/accumulations
```

The calibration daemon and the all-sky monitor use the same image with
`python3 -m streamcal.daemon ...` and `python3 -m skymon run ...`, and with
`--ipc host`.

## Performance

On an imaging node (RTX 4090, driver 550), with the production FX options
(256 x 256 pixels, 192 channels, w-projection, auto-correlations removed),
the calibration tap and the all-sky monitor's image ring:

| Run | Image interval or gulp pacing | Gridding, transform and combination |
|---|---|---|
| unpaced, native build | 24.06 ms per 41.80 ms gulp | 9.83 ms |
| unpaced, image | 24.07 ms per 41.80 ms gulp | 9.81 ms |
| paced at the F-engine rate, whole pipeline, image | 14,160 gulps at 41.80 ms; 0.001 ms mean wait for a free buffer | 9.4 to 10.3 ms |

The image runs at the speed of the native build, at 58 per cent of the gulp
time when unpaced.

## Building locally

The submodules must be checked out (`git submodule update --init --recursive`):

```bash
docker build -t lwa-epic --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) .
```
