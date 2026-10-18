#!/bin/bash
# Build the RfiStage test driver (run on a GPU node with CUDA 12)
set -e
cd "$(dirname "$0")"
GLOG=../../../build-streamcal/src/extern/glog
/usr/local/cuda/bin/nvcc -O2 -std=c++17 -DGLOG_USE_GLOG_EXPORT -arch=sm_89 -I"$GLOG" -I../../../src/extern/glog/src \
  rfi_stage_driver.cu ../../../src/ex/rfi_stage.cu -o rfi_stage_driver \
  -L"$GLOG" -lglog -Xlinker -rpath -Xlinker "$(cd "$GLOG" && pwd)"
