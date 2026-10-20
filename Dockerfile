# EPIC imager, streaming calibration and all-sky monitor in one image.
#
# The image holds epic++ with its libraries, the Python sources (src/python) and
# the Python packages they import. The same image runs every process of an
# imaging node: the imager (epic++), the calibration daemon
# (python -m streamcal.daemon) and the all-sky monitor (python -m skymon run).
#
#   docker build -t lwa-epic .                      # needs the submodules checked out
#   docker run --rm --gpus all lwa-epic ./epic++ --help
#
# Running the imager on an NDP stream needs, from the container runtime:
#   - the GPU (NVIDIA container runtime);
#   - the host's network (multicast) and the RDMA devices /dev/infiniband/*;
#   - the capabilities NET_RAW (the verbs raw-packet queue) and IPC_LOCK
#     (pinned receive buffers) in the container's bounding set. epic++ carries
#     them as file capabilities, so it runs as the unprivileged user epic;
#   - an unlimited memlock limit;
#   - the host's IPC namespace, for the shared-memory rings between the imager,
#     the calibration daemon and the all-sky monitor (/dev/shm).
# See deploy/README.md.

ARG CUDA_VERSION=12.3.2
ARG UBUNTU=ubuntu22.04

########################################################################
# Build: epic++ and the Tensor-Core Correlator
########################################################################
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates curl pkg-config \
        python3.10-dev python3-pip \
        libibverbs-dev librdmacm-dev libpq-dev libcurl4-openssl-dev zlib1g-dev \
        libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

# The Tensor-Core Correlator (ASTRON) and the cudawrappers it is built with
# (1.0.0 for CUDA < 12.4), at the versions the imager was validated with.
ARG TCC_COMMIT=d772f81
ARG CUDAWRAPPERS_VERSION=1.0.0
ARG HUGE_PAGE_ALLOCATOR_COMMIT=6c1f59d96d288bc88014eaa9fee18662bd44b1e6
ARG CUDA_ARCH=89
WORKDIR /opt/src
RUN git clone https://github.com/nlesc-recruit/cudawrappers.git \
    && git -C cudawrappers checkout ${CUDAWRAPPERS_VERSION} \
    && git clone https://git.astron.nl/RD/huge-page-allocator.git \
    && git -C huge-page-allocator checkout ${HUGE_PAGE_ALLOCATOR_COMMIT} \
    && git clone https://git.astron.nl/RD/tensor-core-correlator.git \
    && git -C tensor-core-correlator checkout ${TCC_COMMIT} \
    && cmake -S tensor-core-correlator -B tensor-core-correlator/build \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DTCC_BUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON \
        -DFETCHCONTENT_SOURCE_DIR_CUDAWRAPPERS=/opt/src/cudawrappers \
        -DFETCHCONTENT_SOURCE_DIR_HUGE-PAGE-ALLOCATOR=/opt/src/huge-page-allocator \
    && cmake --build tensor-core-correlator/build -j"$(nproc)"

# epic++. Channels kept from each NDP stream and the verbs receive queue's
# packet buffers (the ConnectX limit on the imaging nodes) are build options.
# The context has no .git: the commit recorded with every image comes from
# GIT_COMMIT (the workflow passes the full SHA).
ARG GIT_COMMIT=unknown
ARG EPIC_NCHAN_SELECT=192
ARG EPIC_VERBS_NPKTBUF=32768
COPY . /opt/src/LWA_EPIC
# cuFFTDx, from NVIDIA MathDx 22.11 (not redistributed in the repository)
ARG MATHDX_URL=https://developer.download.nvidia.com/compute/mathdx/redist/mathdx/linux-x86_64/nvidia-mathdx-22.11.0-Linux.tar.gz
RUN curl -fsSL ${MATHDX_URL} | tar xz --strip-components=1 -C LWA_EPIC/src/extern nvidia-mathdx-22.11.0-Linux/nvidia/mathdx/22.11/include
# NVML is linked against the toolkit's stub; the driver's library is mounted at run time
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
RUN cmake -S LWA_EPIC -B LWA_EPIC/build -DCMAKE_BUILD_TYPE=Release \
        -DPYTHON_EXECUTABLE=/usr/bin/python3.10 \
        -DEPIC_TCC_ROOT=/opt/src/tensor-core-correlator \
        -DEPIC_CUDAWRAPPERS_INCLUDE=/opt/src/cudawrappers/include \
        -DEPIC_NCHAN_SELECT=${EPIC_NCHAN_SELECT} -DEPIC_VERBS_NPKTBUF=${EPIC_VERBS_NPKTBUF} \
        -DEPIC_GIT_COMMIT=${GIT_COMMIT} \
    && cmake --build LWA_EPIC/build --target epic++ -j"$(nproc)"

# epic++ and the shared libraries built with it, laid out for the runtime stage
RUN mkdir -p /out/opt/epic /out/usr/local/lib/epic \
    && cp LWA_EPIC/build/epic++ /out/opt/epic/ \
    && ldd LWA_EPIC/build/epic++ | awk '$3 ~ "^/opt/src/" {print $3}' \
        | xargs -r -I{} cp -L {} /out/usr/local/lib/epic/ \
    && cp -r LWA_EPIC/src/python/. /out/opt/epic/ \
    && cp -r LWA_EPIC/src/commands /out/opt/epic/ \
    && cp LWA_EPIC/version.json /out/opt/epic/ \
    && find /out/opt/epic -name __pycache__ -prune -exec rm -rf {} +

########################################################################
# Python packages, built as wheels (LSL needs FFTW and GSL headers)
########################################################################
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS wheels
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential python3.10-dev python3-pip pkg-config \
        libfftw3-dev libgsl-dev libpq-dev \
    && rm -rf /var/lib/apt/lists/*
# docker/wheelhouse, if present (not in the repository), holds wheels
# downloaded beforehand; pip download skips those and fetches the rest.
COPY docker/ /tmp/docker/
WORKDIR /tmp/docker
RUN mkdir -p wheelhouse \
    && python3.10 -m pip install --no-cache-dir --upgrade pip \
    && python3.10 -m pip download --no-cache-dir --only-binary=:all: -d wheelhouse \
        -r requirements.txt "setuptools>=68" wheel "setuptools-scm[toml]>=6.2" "cython<3" \
    && python3.10 -m pip install --no-cache-dir --no-index --find-links wheelhouse \
        numpy==1.23.5 "setuptools>=68" wheel "setuptools-scm[toml]>=6.2" "cython<3" \
    && python3.10 -m pip wheel --no-cache-dir --no-deps --no-build-isolation -w wheelhouse \
        -r requirements-source.txt \
    && mkdir /wheels && cp wheelhouse/*.whl /wheels/

########################################################################
# Runtime
########################################################################
FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU}

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_VERSION
# The CUDA headers are for NVRTC: the Tensor-Core Correlator compiles its
# kernel when the imager starts, and looks for them beside libnvrtc.
RUN CUDA_PKG=$(echo ${CUDA_VERSION} | cut -d. -f1-2 | tr . -) \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip libpython3.10 \
        libibverbs1 ibverbs-providers librdmacm1 rdma-core ibverbs-utils \
        libpq5 libcurl4 libgomp1 \
        libavcodec58 libavformat58 libavutil56 libswscale5 libavfilter7 \
        libfftw3-double3 libfftw3-single3 libgsl27 libgslcblas0 \
        iproute2 iputils-ping gawk procps libcap2-bin tini \
        cuda-cudart-dev-${CUDA_PKG} cuda-crt-${CUDA_PKG} \
    && rm -rf /var/lib/apt/lists/*

COPY --from=wheels /wheels /tmp/wheels
COPY docker/requirements.txt docker/requirements-source.txt /tmp/
RUN python3.10 -m pip install --no-cache-dir --no-index --find-links /tmp/wheels \
        -r /tmp/requirements.txt \
    && python3.10 -m pip install --no-cache-dir --no-index --no-deps --find-links /tmp/wheels \
        -r /tmp/requirements-source.txt \
    && rm -rf /tmp/wheels /tmp/requirements*.txt

COPY --from=build /out/ /
# cudawrappers loads NVRTC's builtins by their unversioned name, which only
# the toolkit's development package provides; the link is in a default
# directory because a binary with file capabilities ignores LD_LIBRARY_PATH.
RUN ln -s /usr/local/cuda/lib64/libnvrtc-builtins.so.$(echo ${CUDA_VERSION} | cut -d. -f1-2) \
        /usr/lib/x86_64-linux-gnu/libnvrtc-builtins.so \
    && echo /usr/local/lib/epic > /etc/ld.so.conf.d/epic.conf && ldconfig

# The user the processes run as: the owner of the data directories on the
# imaging nodes (uid and gid 1000). Its home holds the astropy and skymon
# caches unless they are mounted from the host.
ARG EPIC_UID=1000
ARG EPIC_GID=1000
RUN groupadd -g ${EPIC_GID} epic && useradd -u ${EPIC_UID} -g epic -m -d /home/epic -s /usr/sbin/nologin epic

# The verbs receiver opens a raw-packet queue (NET_RAW) and pins its buffers
# (IPC_LOCK). As file capabilities they take effect for the unprivileged user,
# provided the container is given them (cap_add); the dynamic loader then
# ignores LD_LIBRARY_PATH, hence ld.so.conf above.
RUN setcap cap_net_raw,cap_ipc_lock+ep /opt/epic/epic++

ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.source="https://github.com/epic-astronomy/LWA_EPIC" \
      org.opencontainers.image.description="EPIC imager, streaming calibration and all-sky monitor" \
      org.opencontainers.image.revision="${GIT_COMMIT}"

ENV PYTHONPATH=/opt/epic \
    PYTHONWARNINGS=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    EPIC_GIT_COMMIT=${GIT_COMMIT}
WORKDIR /opt/epic
USER epic
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["./epic++", "--help"]
