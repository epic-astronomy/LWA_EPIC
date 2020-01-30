####################
CXX           ?= g++
NVCC          ?= nvcc
LINKER        ?= g++
CPPFLAGS      ?=
CXXFLAGS      ?= -O3 -Wall -pedantic
NVCCFLAGS     ?= -O3 -Xcompiler "-Wall" -lineinfo
LDFLAGS       ?=
DOXYGEN       ?= doxygen
PYBUILDFLAGS   ?=
PYINSTALLFLAGS ?=

GPU_ARCHS     ?= 50 # GeForce GTX TITAN X
# Always check the GPU Compute Capability (cc) and compatibility with nvcc version.

# Install may work but the usgae code throws the following error for mismatched GPU and
# cudaGetErrorString(cuda_ret) = no kernel image is available for execution on the device
#  or
# cudaGetErrorString(cuda_ret) = invalid device function

ALIGNMENT ?= 4096 # Memory allocation alignment
GPU_SHAREDMEM ?= 16384 # Fix GPU shared memory per device

CUDA_HOME     ?= /usr/local/cuda-10.2   # check the NVCC version installed and it’s instal directory

CUDA_LIBDIR   ?= $(CUDA_HOME)/lib
CUDA_LIBDIR64 ?= $(CUDA_HOME)/lib64
CUDA_INCDIR   ?= $(CUDA_HOME)/include

ALIGNMENT ?= 4096 # Memory allocation alignment

####################
