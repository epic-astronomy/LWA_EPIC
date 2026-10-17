#!/bin/bash
# Build tcc_correlate against the ASTRON Tensor-Core Correlator (section 8.9 of the notes).
# TCC: git clone https://git.astron.nl/RD/tensor-core-correlator (d772f81) and build it with
#   cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89 \
#         -DFETCHCONTENT_SOURCE_DIR_CUDAWRAPPERS=<cudawrappers 1.0.0 checkout>   # main needs CUDA >= 12.4
#   make -C build
set -e
TCC=${TCC:-/FileStore/epicT/epic/tensor-core-correlator}
CW=${CUDAWRAPPERS:-/FileStore/epicT/epic/cudawrappers-1.0.0}
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -I"$TCC" -I"$CW/include" -I/usr/local/cuda/include tcc_correlate.cc -o tcc_correlate \
    -L"$TCC/build/libtcc" -ltcc -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcuda -lnvrtc \
    -Wl,-rpath,"$TCC/build/libtcc" -Wl,-rpath,/usr/local/cuda/lib64
echo "built $(pwd)/tcc_correlate"
