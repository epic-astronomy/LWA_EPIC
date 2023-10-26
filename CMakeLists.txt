cmake_minimum_required(VERSION 3.10...3.19)
include(CMakePrintHelpers)
include(FetchContent)

project(epic++ VERSION "0.5.1" LANGUAGES CXX CUDA)
set(CMAKE_INSTALL_PREFIX "/home/batman/epic/lib/")
set(PKG_CONFIG_PATH "/home/batman/epic/lib/pkgconfig")
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
OPTION(PYTHON_EXECUTABLE "/home/batman/miniconda3/envs/epic310/bin/python")


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
set(AV_LIBS -lavcodec -lavformat -lavutil -lswscale -lavfilter)

# cmake_print_variables(outVar)
#set(CUDA_NVCC_EXECUTABLE "/home/batman/miniconda3/envs/epic310/bin/nvcc")
#set(CMAKE_CUDA_COMPILER "/home/batman/miniconda3/envs/epic310/bin/nvcc")

add_library(epic_cu STATIC src/MOFF_cu_handler.cu src/ex/host_helpers.cu)
target_compile_features(epic_cu PUBLIC cxx_std_17)
target_link_libraries(epic_cu glog::glog prometheus-cpp::pull)
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


add_dependencies(epic++ raft hwy epiclib)
add_compile_definitions(EPIC_VERSION="${CMAKE_PROJECT_VERSION}")
target_link_libraries(epic++ PRIVATE pybind11::embed raft hwy epiclib epic_cu cudart
glog::glog nvidia-ml pqxx pq ibverbs prometheus-cpp::pull cmake_git_version_tracking ${AV_LIBS}
)

# file(GLOB PY_FILES "src/python/*.py")
# file(COPY ${PY_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E create_symlink
                   ${CMAKE_SOURCE_DIR}/src/python/epic_utils.py $<TARGET_FILE_DIR:${PROJECT_NAME}>/epic_utils.py)

