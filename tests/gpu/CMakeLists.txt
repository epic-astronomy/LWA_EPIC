# Standalone tests and benchmarks for the epic++ imaging kernel.
#
# Builds against the kernel headers in ../../src/ex only (no RaftLib, pybind,
# glog or networking), so it can be built and run on any worker node:
#
#   cmake -S tests/gpu -B build-gpu-tests -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
#   cmake --build build-gpu-tests -j
#   ctest --test-dir build-gpu-tests --output-on-failure
#   build-gpu-tests/bench_imager            # nvbench
#   build-gpu-tests/gulp_budget             # 40 ms budget check
cmake_minimum_required(VERSION 3.30.4)  # nvbench; its rapids-cmake needs CMake >= 4.0
# Must be set before project(), which otherwise picks the compiler default
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)  # cuFFTDx descriptions in fft_dx.cuh use SM<890>
endif()
project(epic_gpu_tests LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(EPIC_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../src" CACHE PATH "epic++ src directory")
set(MATHDX_INCLUDE_DIR "" CACHE PATH "cuFFTDx (MathDx 22.11) include directory")
if(NOT MATHDX_INCLUDE_DIR)
  foreach(candidate
          "${EPIC_SRC_DIR}/extern/nvidia/mathdx/22.11/include"
          "/FileStore/epicT/epic/LWA_EPIC/src/extern/nvidia/mathdx/22.11/include")
    if(EXISTS "${candidate}/cufftdx.hpp")
      set(MATHDX_INCLUDE_DIR "${candidate}")
      break()
    endif()
  endforeach()
endif()
if(NOT EXISTS "${MATHDX_INCLUDE_DIR}/cufftdx.hpp")
  message(FATAL_ERROR "cuFFTDx not found; set -DMATHDX_INCLUDE_DIR=<mathdx>/include")
endif()
message(STATUS "epic++ sources: ${EPIC_SRC_DIR}")
message(STATUS "cuFFTDx: ${MATHDX_INCLUDE_DIR}")

# Same device code generation options as the production epic_cu target
# Same device flags as the production epic_cu target (no --expt-relaxed-constexpr)
set(EPIC_CUDA_FLAGS -O3 -use_fast_math -extra-device-vectorization -lineinfo)

function(epic_gpu_target name src)
  add_executable(${name} ${src})
  target_include_directories(${name} PRIVATE
    "${EPIC_SRC_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_include_directories(${name} SYSTEM PRIVATE "${MATHDX_INCLUDE_DIR}")
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${EPIC_CUDA_FLAGS}>)
endfunction()

epic_gpu_target(test_imager test_imager.cu)
epic_gpu_target(gulp_budget gulp_budget.cu)
epic_gpu_target(kernel_info kernel_info.cu)
epic_gpu_target(image_file image_file.cu)

enable_testing()
add_test(NAME imager_correctness COMMAND test_imager)
add_test(NAME gulp_budget_40ms COMMAND gulp_budget --gulps 100 --budget-ms 40)
set_tests_properties(gulp_budget_40ms PROPERTIES LABELS budget RUN_SERIAL TRUE)

option(EPIC_BUILD_NVBENCH "Build the nvbench benchmarks" ON)
if(EPIC_BUILD_NVBENCH)
  include(FetchContent)
  FetchContent_Declare(nvbench
    GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
    GIT_TAG 410dcdd21c9b48191ecb3d3d77060b1bf4ac6244  # main, 2026-08-31
    GIT_SHALLOW FALSE)
  set(NVBench_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(NVBench_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(nvbench)
  epic_gpu_target(bench_imager bench_imager.cu)
  target_link_libraries(bench_imager PRIVATE nvbench::main)
endif()
