#!/usr/bin/env bash
# Build and run the epic++ GPU kernel tests and benchmarks on a worker node.
#
# usage: tests/gpu/run_on_worker.sh [build dir] [--no-bench]
# Results go to <build dir>/results/<host>-<timestamp>.{txt,json}
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build="${1:-$here/../../build-gpu-tests-$(hostname -s)}"
bench=1
[[ "${2:-}" == "--no-bench" ]] && bench=0

export PATH="/usr/local/cuda/bin:$PATH"

# nvbench (rapids-cmake 25.12) needs CMake >= 4.0; fetch a portable CMake if the system one is older
cmake_bin=cmake
need="4.0.0"
have="$(cmake --version 2>/dev/null | head -1 | awk '{print $3}')"
if [[ -z "$have" || "$(printf '%s\n' "$need" "$have" | sort -V | head -1)" != "$need" ]]; then
  ver=4.0.3
  tools="$here/.tools"
  if [[ ! -x "$tools/cmake-$ver-linux-x86_64/bin/cmake" ]]; then
    mkdir -p "$tools"
    curl -sSL "https://github.com/Kitware/CMake/releases/download/v$ver/cmake-$ver-linux-x86_64.tar.gz" | tar -xz -C "$tools"
  fi
  cmake_bin="$tools/cmake-$ver-linux-x86_64/bin/cmake"
fi

"$cmake_bin" -S "$here" -B "$build" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc > /dev/null
"$cmake_bin" --build "$build" -j "$(nproc)"

stamp="$(hostname -s)-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$build/results"
log="$build/results/$stamp.txt"
{
  echo "host: $(hostname -s)  date: $(date -u)  commit: $(git -C "$here" rev-parse --short HEAD)"
  nvidia-smi --query-gpu=name,compute_cap,driver_version,clocks.max.sm,temperature.gpu --format=csv,noheader
  echo
  echo "== correctness =="
  "$build/test_imager"
  echo
  echo "== gulp budget (production settings, 40 ms) =="
  "$build/gulp_budget" --gulps 200 --budget-ms 40
  if [[ $bench == 1 ]]; then
    echo
    echo "== nvbench =="
    "$build/bench_imager" --json "$build/results/$stamp.json"
  fi
} 2>&1 | tee "$log"
echo "results: $log"
