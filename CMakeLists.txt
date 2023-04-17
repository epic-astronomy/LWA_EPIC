cmake_minimum_required(VERSION 3.10...3.19)
include(CMakePrintHelpers)


project(epic++ VERSION "0.0.1" LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)



### Raftlib
include_directories(src/ex)
include_directories(src/extern/RaftLib/raftinc)
include_directories(src/extern/RaftLib/git-dep/shm/include)
include_directories(src/extern/RaftLib/git-dep/highwayhash/include)
include_directories(src/extern/RaftLib/git-dep/demangle/include)
include_directories(src/extern/RaftLib/git-dep/cmdargs/include)
include_directories(src/extern/RaftLib/git-dep/affinity/include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#include_directories(/usr/local/cuda/targets/x86_64-linux/include)

cmake_print_variables(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)


set(BUILD_EXAMPLES  False CACHE BOOL "Disable building raft's examples")
set(BUILD_BENCHMARKS False CACHE BOOL "Disable raft's benchmarks")
set(BUILD_TESTS False CACHE BOOL "Disable raft's tests")
add_subdirectory(src/extern/RaftLib)
add_compile_options(-std=c++17  -DL1D_CACHE_LINE_SIZE=64 -DSTRING_NAMES=1)

#glog
add_subdirectory(src/extern/glog)

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

include_directories(src/extern/highway)

### pybind11
add_subdirectory(src/extern/pybind11)
OPTION(PYTHON_EXECUTABLE "/home/epic/anaconda/envs/esp38/bin/python")

# include_directories(src/extern/glog/src)

### cuFFTDx
include_directories(src/extern/nvidia/mathdx/22.11/include)

# include_directories(src/extern/thread_pool)

find_package(OpenMP REQUIRED)


# although the current EPIC system supports AVX512, each channel contains 256 bits
# so setting the max vector size to 256 bits can give optimal performance
add_compile_options(-march=haswell -maes -O3 -pthread -w)
add_library(epiclib
   src/sockets.cpp
)

# execute_process (
#     COMMAND bash -c "nvcc -V"
#     OUTPUT_VARIABLE outVar
# )

# cmake_print_variables(outVar)
set(CUDA_NVCC_EXECUTABLE "/usr/local/cuda-11.6/bin/nvcc")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.6/bin/nvcc")

add_library(epic_cu STATIC src/MOFF_cu_handler.cu src/ex/host_helpers.cu)
target_compile_features(epic_cu PUBLIC cxx_std_17)
target_compile_options(epic_cu PRIVATE -O3 -use_fast_math )
# add_dependencies(epic_cu glog)
# target_include_directories(epic_cu PRIVATE src/extern/glog)
# set(CUDA_INCLUDE_DIRS src/extern/glog)
set_target_properties( epic_cu
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(epic_cu PROPERTIES CUDA_ARCHITECTURES "86")


add_executable(epic++ src/main.cpp)
set_target_properties( epic++
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_dependencies(epic++ raft hwy epiclib)
add_compile_definitions(EPIC_VERSION="${CMAKE_PROJECT_VERSION}")
target_link_libraries(epic++ PRIVATE pybind11::embed raft hwy epiclib
ibverbs epic_cu cudart
glog::glog
#OpenMP::OpenMP_CXX
)

file(GLOB PY_FILES "src/python/*.py")
file(COPY ${PY_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
# configure_file(${files} ${CMAKE_CURRENT_BINARY_DIR}/epic_utils.py COPYONLY)




# add_library(pkt_assembler
#    src/packet_assembler.cpp
# )

# add_dependencies(pkt_assembler epiclib)
# target_link_libraries(pkt_assembler epiclib)