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

GPU_ARCHS     ?= 50 # GeForce GTX TITAN X, change if you are using a different card.
# Installation with wrong GPU_ARCHS above results in an error like:
# cudaGetErrorString(cuda_ret) = no kernel image is available for execution on the device
#  or
# cudaGetErrorString(cuda_ret) = invalid device function

ALIGNMENT ?= 4096 # Memory allocation alignment
GPU_SHAREDMEM ?= 16384 # Fix GPU shared memory per device

# check the NVCC version installed and its install directory
CUDA_HOME     ?= /usr/local/cuda-11.4

CUDA_LIBDIR   ?= $(CUDA_HOME)/lib
CUDA_LIBDIR64 ?= $(CUDA_HOME)/lib64
CUDA_INCDIR   ?= $(CUDA_HOME)/include

ALIGNMENT ?= 4096 # Memory allocation alignment

####################
