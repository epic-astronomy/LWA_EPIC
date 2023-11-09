/*
 Copyright (c) 2023 The EPIC++ authors

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

#ifndef SRC_EX_CU_HELPERS_CUH_
#define SRC_EX_CU_HELPERS_CUH_

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cassert>
#include <cufftdx.hpp>
#include <iostream>
#include <stdexcept>

#include "./constants.h"
#include "./host_helpers.h"
#include "./types.hpp"

namespace cg = cooperative_groups;

/**
 * @brief Half-precision complex multipy scale
 *
 * Multiply two complex numbers and optinally scale it with a scalar. The
 * multiplication is an fma operation and uses the optimized half2 intrinsics.
 *
 * @param a First input complex value
 * @param b Second input complex value
 * @param scale
 * @return s* (A * B)
 */
__device__ __half2 __half2cms(__half2 a, __half2 b, __half scale = __half(1)) {
  return __hmul2(__halves2half2(scale, scale),
                 __halves2half2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x));
}

/**
 * @brief Mixed-precision complex multiply and scale using built-in intrinsics
 *
 * Performs s*(A * B), where s is a scalar, A and B are two complex numbers
 *
 * @tparam T Type of output complex data
 * @param out[out] Output variable
 * @param a[in] Complex number of type float2
 * @param b[in] Complex number of type CNib
 * @param scale[in] Scale valie
 * @return void
 *
 * @relatesalso MOFFCuHandler
 */
template <typename T>
__device__ void __cms_f(T& out, const float2 a, const CNib b, float& scale) {
  out.x =
      __fmul_rz(scale, __fadd_rz(__fmul_rz(a.x, b.re), -__fmul_rz(a.y, b.im)));
  out.y =
      __fmul_rz(scale, __fadd_rz(__fmul_rz(a.x, b.im), __fmul_rz(a.y, b.re)));
}

/**
 * @brief Mixed-precision complex multiply using built-in intrinsics
 *
 *
 * @tparam T Type of the output complex data
 * @param out[out] Output variable
 * @param a[in] Complex value of type float2
 * @param b[in] Complex value of type CNib
 * @return void
 *
 * @relatesalso MOFFCuHandler
 */
template <typename T>
__device__ void __cm_f(T& out, const float2& a, const CNib& b) {
  out.x += __fadd_rz(__fmul_rz(a.x, b.re), -__fmul_rz(a.y, b.im));
  out.y += __fadd_rz(__fmul_rz(a.x, b.im), __fmul_rz(a.y, b.re));
}

/**
 * @brief Get a const pointer to a specific sequence from F-engine data
 *
 * @tparam Order Order of the data
 * @param f_eng Pointer to F-Engine data
 * @param gulp_idx 0-based time-index of the gulp
 * @param chan_idx Channel number of the sequence
 * @param ngulps Total number sequences in the gulp
 * @param nchan Total number of channels in each sequence
 * @return Pointer to the start of the sequence
 *
 * @relatesalso MOFFCuHandler
 */
template <PKT_DATA_ORDER Order>
__device__ const uint8_t* GetFEngSample(const uint8_t* f_eng,
                                           size_t gulp_idx, size_t chan_idx,
                                           size_t nseqs, size_t nchan) {
  if (Order == CHAN_MAJOR) {
    return f_eng + LWA_SV_INP_PER_CHAN * (chan_idx * nseqs + gulp_idx);
  }
  if (Order == TIME_MAJOR) {
    return f_eng + LWA_SV_INP_PER_CHAN * (gulp_idx * nchan + chan_idx);
  }
}

/**
 * @brief Get antenna positions for a given channel
 *
 * The antenna position array must have dimensions nchan, nant, 3
 *
 * @param ant_pos Pointer to the antenna position array
 * @param chan_idx 0-based channel index
 * @return Pointer to the antenna position sub-array
 *
 * @relatesalso MOFFCuHandler
 */
__device__ const float* GetAntPos(const float* ant_pos, size_t chan_idx) {
  return ant_pos + chan_idx * LWA_SV_NSTANDS * 3 /*dimensions*/;
}

/**
 * @brief Get pointer to antenna phases for the specified channel
 *
 * Phases array must have dimensions nchan, nant, 2
 *
 * @param phases Pointer to the phases array
 * @param chan_idx 0-based channel index
 * @return const pointer to the phases array for `chan_idx`
 *
 * @relatesalso MOFFCuHandler
 */
__device__ const float* GetPhases(const float* phases, size_t chan_idx) {
  return phases + chan_idx * LWA_SV_NSTANDS * 2 /*phases*/ * 2 /*real imag*/;
};

/**
 * @brief Transpose a matrix by swapping its upper and lower triangles
 *
 *
 * @tparam FFT FFT built using cuFFTDx
 * @param thread_reg Thread registers
 * @param shared_mem Shared memory. Must have at least half the memory size of
 * the matrix
 * @param norm Normalizing factor
 * @return __device__
 *
 * @note Although this transpose takes half the memory of the original matrix,
 * it also requires twice the number of shared mem reads and writes compared to
 * a full transpose within the shared memory.
 */
template <
    class FFT,
    std::enable_if_t<
        std::is_same<__half2, typename FFT::output_type::value_type>::value,
        bool> = true>
__device__ void TransposeTri(
    typename FFT::value_type (&thread_reg)[FFT::elements_per_thread],
    typename FFT::value_type* shared_mem,
    typename cufftdx::precision_of<FFT>::type norm = 1.) {
  constexpr int stride =
      cufftdx::size_of<FFT>::value / FFT::elements_per_thread;

  // copy the upper triangle into the shared memory
  for (int i = 0; i < FFT::elements_per_thread; ++i) {
    auto col = threadIdx.x + i * stride;
    if (col <= threadIdx.y) {
      continue;
    }
    // The index for the upper triangle element in the shared memory is the
    // original index offset by the left triangle and diagonal elements.
    int upper_smem_idx = threadIdx.y * cufftdx::size_of<FFT>::value + col -
                         (threadIdx.y + 1) * (threadIdx.y + 2) * 0.5;
    shared_mem[upper_smem_idx] = thread_reg[i];
  }

  __syncthreads();

  // swap the elements in lower traingle with those in the shared memory
  for (int i = 0; i < FFT::elements_per_thread; ++i) {
    auto col = threadIdx.x + i * stride;
    if (threadIdx.y <= col) {
      continue;
    }
    // The index for the lower triangle element is the same as the upper
    // triangle but with the row and columns exchanged.
    int lower_smem_idx = col * cufftdx::size_of<FFT>::value + threadIdx.y -
                         (col + 1) * (col + 2) * 0.5;
    auto _temp = thread_reg[i];
    thread_reg[i] = shared_mem[lower_smem_idx];
    shared_mem[lower_smem_idx] = _temp;
  }

  __syncthreads();

  // copy the lower traiangle into the upper triangle
  for (int i = 0; i < FFT::elements_per_thread; ++i) {
    auto col = threadIdx.x + i * stride;
    if (col <= threadIdx.y) {
      continue;
    }
    int upper_smem_idx = threadIdx.y * cufftdx::size_of<FFT>::value + col -
                         (threadIdx.y + 1) * (threadIdx.y + 2) * 0.5;
    thread_reg[i] = shared_mem[upper_smem_idx];
  }

  // normalize
  if (float(norm) != 1) {
    for (int i = 0; i < FFT::elements_per_thread; ++i) {
      thread_reg[i].x *= __half2half2(norm);
      thread_reg[i].y *= __half2half2(norm);
    }
  }
};

/**
 * @brief Perform Riemann integration of the GCF over a pixel
 *
 * @param gcf_tex GCF texture object
 * @param dx X-component of the vector between the antenna and the lower-left
 * corner of the pixel (pix_x - antx).
 * @param dy  Y-component of the vector between the antenna and the lower-left
 * corner of the pixel (pix_y - anty).
 * @param nsteps Number of integration steps. Defaults to 5.
 * @return float
 */
__device__ float GcfPixelIntegral(cudaTextureObject_t gcf_tex, float dx,
                                    float dy, unsigned int d_per_pixel,
                                    int nsteps = 5) {
  float sum = 0;
  float delta = 1.f / float(nsteps);
  float offset =
      1.f /
      float(2 *
            nsteps);  // nudges the texture point to the center of each dxy cell

#pragma unroll
  for (float x = offset; x < 1; x += delta) {
    for (float y = offset; y < 1; y += delta) {
      sum += tex2D<float>(gcf_tex, abs(dx + x) * d_per_pixel,
                          abs(dy + y) * d_per_pixel);
    }
  }

  return sum;
}

/**
 * @brief Compute gcf kernel cell scaling values for each antenna and frequency.
 *
 * The gcf is integrated over each cell to estimate the scaling value and are
 * normalized to 1. This ensures flux conservation even if the antenna
 * illumination pattern does not fit within the kernel dimensions.
 *
 * @param out
 * @param antpos
 * @param chan0
 * @param lmbda_scale
 * @param gcf_tex
 * @param grid_size
 * @param support
 * @param nants
 * @return
 */
__global__ void ComputeGcfElements(float* out, float* antpos, int chan0,
                                     float lmbda_scale,
                                     cudaTextureObject_t gcf_tex,
                                     int grid_size /*unused*/, int support = 3,
                                     int nants = LWA_SV_NSTANDS) {
  int nelements = support * support;
  int half_support = support / 2;
  auto tb = cg::this_thread_block();
  int grp_rank = tb.thread_rank() / nelements;
  // int offset = 0;
  int grp_size = tb.num_threads() / nelements;
  int grp_thread_rank = tb.thread_rank() % nelements;

  extern __shared__ float antenna_sum[];

  if (tb.thread_rank() < nants) {
    antenna_sum[tb.thread_rank()] = 0;
  }

  assert(tb.size() % nelements == 0);

  __syncthreads();
  int channel_idx = blockIdx.x;

  auto* antpos_chan =
      reinterpret_cast<const float3*>(GetAntPos(antpos, channel_idx));

  float dist_scale =
      float(SOL) / float((channel_idx + chan0) * BANDWIDTH) * lmbda_scale * 10.; //decimeters

  for (int ant = grp_rank; ant < nants; ant += grp_size) {
    int dy = half_support - (grp_thread_rank) / (support);
    int dx = (grp_thread_rank) % (support)-half_support;

    float antx = antpos_chan[ant].x;
    float anty = antpos_chan[ant].y;

    // antx=int(antx)+0.5;
    // anty=int(anty)+0.5;

    int xpix = int(antx + dx);
    int ypix = int(anty + dy);

    bool is_pix_valid = true;
    if (xpix < 0 || xpix >= grid_size || ypix < 0 || ypix >= grid_size) {
      is_pix_valid = false;
    }

    float integral = is_pix_valid
                         ? GcfPixelIntegral(gcf_tex, (xpix - antx),
                                              (ypix - anty), dist_scale)
                         : 0;

    atomicAdd(&antenna_sum[ant], integral);
    tb.sync();

    float norm = antenna_sum[ant] != 0 ? 1.f / antenna_sum[ant] : 1.0;
    integral *= norm;
    // if(blockIdx.x==9 && threadIdx.x<49 && ant==0){
    //     printf("%d %d %f\n",dx, dy,  integral);
    // }

    out[channel_idx * nants * nelements + ant * nelements + grp_thread_rank] =
        integral;

    tb.sync();
  }
}

/**
 * @brief Compute the antenna-averaged kernel for each frequency
 *
 * @param grid_elems Individual anetenna gridding kernels stored in a row-major
 * order
 * @param[out] out_kernel Output gridding kernels
 * @param nchan Number of channels
 * @param support_size Support size of the kernel
 * @return
 */
__global__ void ComputeAvgGriddingKernel(float* grid_elems,
                                            float* out_kernel, int nchan,
                                            int support_size) {
  int nelems_per_ant = support_size * support_size;
  int ant = threadIdx.x;
  int chan = blockIdx.x;
  int nelemns_per_chan = nelems_per_ant * blockDim.x;
  int grid_idx = ant * nelems_per_ant + chan * nelemns_per_chan;

  // each thread adds all elements of one antenna to the output grid
  for (int i = 0; i < nelems_per_ant; ++i) {
    atomicAdd(&out_kernel[blockIdx.x * nelems_per_ant + i],
              grid_elems[grid_idx + i]);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    float sum = 0;
    for (int i = 0; i < nelems_per_ant; ++i) {
      sum += out_kernel[blockIdx.x * nelems_per_ant + i];
    }

    for (int i = 0; i < nelems_per_ant; ++i) {
      out_kernel[blockIdx.x * nelems_per_ant + i] /= sum;
    }
  }
}

inline __host__ __device__ float4 operator+(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __host__ __device__ void operator+=(float4& a, float4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}

inline __host__ __device__ void operator+=(float2& a, float2 b) {
  a.x += b.x;
  a.y += b.y;
}

inline __host__ __device__ void operator+=(float3& a, float3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

inline __host__ __device__ float4 make_float4(__nv_bfloat162 a,
                                              __nv_bfloat162 b) {
  return make_float4(float(a.x), float(a.y), float(b.x), float(b.y));
}

inline __host__ __device__ float4 make_float4_s(__nv_bfloat162& a,
                                                __nv_bfloat162& b,
                                                float scale) {
  return float4{float(a.x) * scale, float(a.y) * scale, float(b.x) * scale,
                float(b.y) * scale};
}

template <typename In, class Outv = float4>
inline __host__ __device__ Outv make_v4_s(In& a, In& b, float scale) {
  using Out = decltype(Outv::x);
  return Outv{Out(a.x) * scale, Out(a.y) * scale, Out(b.x) * scale,
              Out(b.y) * scale};
}

inline __host__ __device__ float2 operator*(float2 a, float2 b) {
  return make_float2(a.x * b.x, a.y * b.y);
}

inline __host__ __device__ float2 operator*(float2 a, float b) {
  return make_float2(a.x * b, a.y * b);
}

inline __host__ __device__ float3 operator*(float3 a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ float4 operator*(float4 a, float4 b) {
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

inline __host__ __device__ float2 operator+(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float4 operator*(float4 a, float b) {
  return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

/// @brief Compute X*X.conj()
/// @tparam T Output data type
/// @tparam FFT FFT object constructed using cuFFTDx
/// @param pix Complex pixel value
/// @return T
template <
    typename T, class FFT,
    std::enable_if_t<
        std::is_same<__half2, typename FFT::output_type::value_type>::value,
        bool> = true>
inline __host__ __device__ T
ComputeXX(typename FFT::value_type
               pix) {  // half precision intrinsics reduce the performance here
  // return T(__hfma(pix.x.x, pix.x.x, pix.y.x * pix.y.x));
  return static_cast<T>(pix.x.x * pix.x.x + pix.y.x * pix.y.x);
}

/**
 * @brief Compute Y*Y.conj()
 *
 * @tparam T Output data type
 * @tparam FFT FFT object constructed using cuFFTDx
 * @param pix Complex pixel value
 * @return T
 */
template <
    typename T, class FFT,
    std::enable_if_t<
        std::is_same<__half2, typename FFT::output_type::value_type>::value,
        bool> = true>
inline __host__ __device__ T ComputeYY(typename FFT::value_type pix) {
  return static_cast<T>(pix.x.y * pix.x.y + pix.y.y * pix.y.y);
}

/**
 * @brief Compute Re(X*Y.conj())
 *
 * @tparam T Output data type
 * @tparam FFT FFT object constructed using cuFFTDx
 * @param pix Complex pixel value
 * @return T
 */
template <
    typename T, class FFT,
    std::enable_if_t<
        std::is_same<__half2, typename FFT::output_type::value_type>::value,
        bool> = true>
inline __host__ __device__ T ComputeUU(typename FFT::value_type pix) {
  return static_cast<T>(pix.x.x * pix.y.x + pix.x.y * pix.y.y);
}

/**
 * @brief Compute Im(X*Y.conj())
 *
 * @tparam T Output data type
 * @tparam FFT FFT object constructed using cuFFTDx
 * @param pix Complex pixel value
 * @return T
 */
template <
    typename T, class FFT,
    std::enable_if_t<
        std::is_same<__half2, typename FFT::output_type::value_type>::value,
        bool> = true>
inline __host__ __device__ T ComputeVV(typename FFT::value_type pix) {
  return static_cast<T>(pix.x.y * pix.y.x - pix.x.x * pix.y.y);
}

#endif  // SRC_EX_CU_HELPERS_CUH_
