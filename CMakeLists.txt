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


### Google benchmark
set(BENCHMARK_ENABLE_TESTING off CACHE BOOL "Disable GBenchmark's testing")
set(BENCHMARK_DOWNLOAD_DEPENDENCIES off CACHE BOOL "Disable GBenchmark's testing")
set(BENCHMARK_USE_BUNDLED_GTEST off CACHE BOOL "Disable GBenchmark's testing")
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
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

### cuFFTDx
include_directories(src/extern/nvidia/mathdx/22.11/include)

include_directories(src/extern/thread_pool)

find_package(OpenMP REQUIRED)


# although the current EPIC system supports AVX512, each channel contains 256 bits
# so setting the max vector size to 256 bits can give optimal performance
add_compile_options(-march=haswell -maes -O3 -pthread -w) 
add_library(epiclib
   src/sockets.cpp 
)

execute_process (
    COMMAND bash -c "nvcc -V"
    OUTPUT_VARIABLE outVar
)

cmake_print_variables(outVar)


add_library(epic_cu STATIC src/MOFF_cu_handler.cu src/ex/host_helpers.cu)
target_compile_features(epic_cu PUBLIC cxx_std_17)
target_compile_options(epic_cu PRIVATE -O3 -use_fast_math -Xptxas -v) 
set_target_properties( epic_cu
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(epic_cu PROPERTIES CUDA_ARCHITECTURES "86")


add_executable(epic++ src/main.cpp)
set_target_properties( epic++
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  

add_dependencies(epic++ raft hwy epiclib)
target_link_libraries(epic++ PRIVATE pybind11::embed raft hwy epiclib ibverbs epic_cu cudart 
#OpenMP::OpenMP_CXX
)




# add_library(pkt_assembler
#    src/packet_assembler.cpp
# )

# add_dependencies(pkt_assembler epiclib)
# target_link_libraries(pkt_assembler epiclib)