cmake_minimum_required(VERSION 3.10...3.19)
include(CMakePrintHelpers)
include(FetchContent)

project(epic++ LANGUAGES CXX CUDA)
#set(CMAKE_INSTALL_PREFIX "/home/batman/epic/lib/")
#set(PKG_CONFIG_PATH "/home/batman/epic/lib/pkgconfig")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "") 

# set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif()

### pqxx
add_subdirectory(src/extern/pqxx)
include_directories(SYSTEM src/extern/pqxx/include)

### Raftlib
include_directories(SYSTEM src/ex)
include_directories(SYSTEM src/extern/RaftLib/raftinc)
include_directories(SYSTEM src/extern/RaftLib/git-dep/shm/include)
include_directories(SYSTEM src/extern/RaftLib/git-dep/highwayhash/include)
include_directories(SYSTEM src/extern/RaftLib/git-dep/demangle/include)
include_directories(SYSTEM src/extern/RaftLib/git-dep/cmdargs/include)
include_directories(SYSTEM src/extern/RaftLib/git-dep/affinity/include)
include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#include_directories(/usr/local/cuda/targets/x86_64-linux/include)

cmake_print_variables(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)


set(BUILD_EXAMPLES  False CACHE BOOL "Disable building raft's examples")
set(BUILD_BENCHMARKS False CACHE BOOL "Disable raft's benchmarks")
set(BUILD_TESTS False CACHE BOOL "Disable raft's tests")
add_subdirectory(src/extern/RaftLib)
add_compile_options(-std=c++17  -DL1D_CACHE_LINE_SIZE=64 -DSTRING_NAMES=1 )


#glog
add_definitions(-DGLOG_NO_SYMBOLIZE)
set(WITH_UNWIND off CACHE BOOL "Disable unwind" FORCE)
add_subdirectory(src/extern/glog)



#cxxopts
add_subdirectory(src/extern/cxxopts)
include_directories(SYSTEM src/extern/cxxopts/include)

### Google benchmark
set(BENCHMARK_ENABLE_TESTING off CACHE BOOL "Disable GBenchmark's testing" FORCE)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES off CACHE BOOL "Disable GBenchmark's testing" FORCE)
set(BENCHMARK_USE_BUNDLED_GTEST off CACHE BOOL "Disable GBenchmark's testing" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable GBenchmark's testing" FORCE)
add_subdirectory(src/extern/benchmark)


### Google highway
set(HWY_ENABLE_TESTS False CACHE BOOL "Disable hwy tests" FORCE)
set(HWY_ENABLE_EXAMPLES False CACHE BOOL "Disable hwy examples" FORCE)
set(HWY_ENABLE_CONTRIB False CACHE BOOL "Disable hwy examples" FORCE)
add_subdirectory(src/extern/highway)

include_directories(SYSTEM src/extern/highway)

### pybind11
add_subdirectory(src/extern/pybind11)
#OPTION(PYTHON_EXECUTABLE "/home/epic/anaconda/envs/epic310/bin/python")


### cuFFTDx
include_directories(SYSTEM src/extern/nvidia/mathdx/22.11/include)

### prometheus-cpp
add_subdirectory(src/extern/prometheus-cpp)
# sudo apt install libcurl4-openssl-dev

find_package(OpenMP REQUIRED)
FetchContent_Declare(cmake_git_version_tracking                   
  GIT_REPOSITORY https://github.com/andrew-hardin/cmake-git-version-tracking.git
  GIT_TAG 904dbda1336ba4b9a1415a68d5f203f576b696bb
)
FetchContent_MakeAvailable(cmake_git_version_tracking)



add_library(epiclib STATIC
   src/sockets.cpp
)

#install libavcodec-dev libswscale-dev libavfilter-dev
set(AV_LIBS -lavcodec -lavformat -lavutil -lswscale -lavfilter )

# cmake_print_variables(outVar)
#set(CUDA_NVCC_EXECUTABLE "/home/batman/miniconda3/envs/epic310/bin/nvcc")
#set(CMAKE_CUDA_COMPILER "/home/batman/miniconda3/envs/epic310/bin/nvcc")

add_library(epic_cu STATIC src/MOFF_cu_handler.cu src/ex/host_helpers.cu src/ex/tcc_tap.cu)
target_compile_features(epic_cu PUBLIC cxx_std_17)
target_link_libraries(epic_cu glog::glog prometheus-cpp::pull)

# Optional GPU calibration tap (--caltap_gpu) with the ASTRON Tensor-Core
# Correlator: EPIC_TCC_ROOT is a built checkout of
# https://git.astron.nl/RD/tensor-core-correlator (libtcc in build/libtcc) and
# EPIC_CUDAWRAPPERS_INCLUDE the include directory of the cudawrappers it was
# built with (1.0.0 for CUDA < 12.4).
set(EPIC_TCC_ROOT "" CACHE PATH "Tensor-Core Correlator checkout (empty: no --caltap_gpu)")
set(EPIC_CUDAWRAPPERS_INCLUDE "" CACHE PATH "cudawrappers include directory")
if(EPIC_TCC_ROOT)
  find_library(EPIC_TCC_LIB tcc PATHS ${EPIC_TCC_ROOT}/build/libtcc NO_DEFAULT_PATH REQUIRED)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(epic_cu PRIVATE EPIC_WITH_TCC)
  target_include_directories(epic_cu PRIVATE ${EPIC_TCC_ROOT} ${EPIC_CUDAWRAPPERS_INCLUDE})
  target_link_libraries(epic_cu ${EPIC_TCC_LIB} CUDA::cuda_driver CUDA::nvrtc)
  message(STATUS "GPU calibration tap: ${EPIC_TCC_LIB}")
endif()
target_compile_options(epic_cu PRIVATE -O3 -use_fast_math --ptxas-options=-v  -Xptxas -v -extra-device-vectorization -Wall)

set_target_properties( epic_cu
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(epic_cu PROPERTIES CUDA_ARCHITECTURES "89")


add_executable(epic++ src/main.cpp)
# although the current EPIC system supports AVX512, each channel contains 256 bits
# so setting the max vector size to 256 bits can give optimal performance
target_compile_options(epic++ PRIVATE -march=haswell -maes -O3 -pthread  -Wall -Wextra -Wpedantic -fconcepts)

set_target_properties( epic++
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# The device link of epic_cu's relocatable kernels happens here, so it needs
# the same architecture; otherwise it uses the CMAKE_CUDA_ARCHITECTURES default
# (52) and every kernel launch fails with "no kernel image is available".
set_target_properties(epic++ PROPERTIES CUDA_ARCHITECTURES "89")


add_dependencies(epic++ raft hwy epiclib)
#add_compile_definitions(EPIC_VERSION="${CMAKE_PROJECT_VERSION}")
target_link_libraries(epic++ PRIVATE OpenMP::OpenMP_CXX pybind11::embed raft hwy epiclib epic_cu cudart
glog::glog nvidia-ml pqxx pq ibverbs prometheus-cpp::pull cmake_git_version_tracking ${AV_LIBS}  
)

# file(GLOB PY_FILES "src/python/*.py")
# file(COPY ${PY_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E create_symlink
                   ${CMAKE_SOURCE_DIR}/src/python/epic_utils.py $<TARGET_FILE_DIR:${PROJECT_NAME}>/epic_utils.py)
# epic_utils imports epic_geometry (phase centre, image WCS, w-projected kernels)
add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E create_symlink
                   ${CMAKE_SOURCE_DIR}/src/python/epic_geometry.py $<TARGET_FILE_DIR:${PROJECT_NAME}>/epic_geometry.py)

