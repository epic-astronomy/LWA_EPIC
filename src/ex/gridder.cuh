#pragma once

#include "constants.h"
#include "cu_helpers.cuh"
#include "types.hpp"
#include <cooperative_groups.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <cufftdx.hpp>
#include <unistd.h>

namespace cg = cooperative_groups;
using namespace cufftdx;

#define WARP_MASK 0xFFFFFFFF

/**
 * @brief Grid the F-Engine data
 *
 * The thread block is tiled with a size of \p Support x \p Support and each
 * tile computes the grid elements of a single antenna at a time. A grid is
 * maintained in the shared memory and the grid elements are atomically added to
 * the grid. This grid only works with half precision.
 *
 * @tparam FFT FFT object constructed using cuFFTDx
 * @tparam Support Support Size. Must be a power of 2 for optimal performance
 * @tparam NStands Number of antennas
 * @param tb Thread block object
 * @param thread_data Thread register array for use as temporary workspace
 * @param f_eng Pointer to the F-Engine data to be gridded
 * @param antpos Pointer to antenna position array
 * @param phases Pointer to the phases array
 * @param smem Pointer to shared memory
 * @param gcf_tex GCF texture object
 * @return void
 *
 * @relatesalso MOFFCuHandler
 */
template <
    typename FFT, unsigned int Support, unsigned int NStands,
    std::enable_if_t<
        std::is_same<__half2, typename FFT::output_type::value_type>::value,
        bool> = true>
__device__ inline void grid_dual_pol_dx5(
    cg::thread_block tb,
    typename FFT::value_type thread_data[FFT::elements_per_thread],
    const cnib2 *f_eng, const float3 *__restrict__ antpos,
    const float4 *__restrict__ phases, typename FFT::value_type *smem,
    cudaTextureObject_t gcf_tex) {
  using complex_type = typename FFT::value_type;
  constexpr float half_support = Support / 2.f;
  constexpr float inv_support = 1.f / float(Support);
  constexpr float inv_half_support = 1.f / float(half_support);
  auto tile = cg::tiled_partition<Support * Support>(tb);

  for (int ant = tile.meta_group_rank(); ant < NStands;
       ant += tile.meta_group_size()) {
    float scale = 1;
    thread_data[0] = __half2half2(0);
    float antx = antpos[ant].x;
    float anty = antpos[ant].y;

    float v = int(tile.thread_rank() * inv_support) - (half_support) + 0.5;
    float u = tile.thread_rank() -
              int(tile.thread_rank() * inv_support) * Support - (half_support) +
              0.5;

    if (!((0 <= int(antx + u) <= size_of<FFT>::value) &&
          (0 <= int(anty + v) <= size_of<FFT>::value) &&
          abs(int(antx + u) + 0.5 - antx) < half_support &&
          abs(int(anty + v) + 0.5 - anty) < half_support)) {
    } else {
      auto phase_ant = phases[ant];
      scale = tex2D<float>(
          gcf_tex, abs((int(antx + u) + 0.5 - antx) * inv_half_support),
          abs((int(anty + v) + 0.5 - anty) * inv_half_support));

      __cms_f(thread_data[0].x, float2{phase_ant.x, phase_ant.y}, f_eng[ant].X,
              scale);
      __cms_f(thread_data[0].y, float2{phase_ant.z, phase_ant.w}, f_eng[ant].Y,
              scale);

      thread_data[1] = thread_data[0];
      thread_data[0].x.y = thread_data[1].y.x;
      thread_data[0].y.x = thread_data[1].x.y;

      atomicAdd(&smem[int(antx + u) + int(anty + v) * size_of<FFT>::value].x,
                thread_data[0].x);
      atomicAdd(&smem[int(antx + u) + int(anty + v) * size_of<FFT>::value].y,
                thread_data[0].y);
    }
  }
}

/**
 * @brief On-chip memory based gridding.
 *
 * The gridded data is written directly to the thread registers without
 * requiring any atomic operations. * For GCF kernel sizes less than the stride
 * of the FFT, each antenna can atmost contribute to one pixel (or a register).
 * The gridder loops through all antennas in each thread and write gridded
 * values to thr appropriate registers.
 *
 * @tparam FFT FFT object constructed using cuFFTDx
 * @tparam Support Support Size
 * @tparam NStands Number of antennas
 * @param thread_data Thread register array for use as temporary workspace
 * @param f_eng Pointer to the F-Engine data to be gridded
 * @param antpos Pointer to antenna position array
 * @param phases Pointer to the phases array
 * @param gcf_tex
 * @return __device__
 */
template <
    typename FFT, unsigned int Support, unsigned int NStands,
    std::enable_if_t<
        std::is_same<__half2, typename FFT::output_type::value_type>::value,
        bool> = true>
__device__ inline void grid_dual_pol_dx6(
    typename FFT::value_type thread_data[FFT::elements_per_thread],
    const cnib2 *f_eng, const float3 *__restrict__ antpos,
    const float4 *__restrict__ phases, cudaTextureObject_t gcf_tex) {
  constexpr float half_support = Support / 2.f;
  constexpr float inv_half_support = 2. / float(Support);
  constexpr int stride = float(size_of<FFT>::value) / FFT::elements_per_thread;
  constexpr double inv_stride = 1. / double(stride);
  static_assert(Support < size_of<FFT>::value / FFT::elements_per_thread,
                "Support size must be less than the stride");

  for (int ant = 0; ant < NStands; ++ant) {
    float antx = antpos[ant].x;
    float anty = antpos[ant].y;

    if (abs(anty - (threadIdx.y + 0.5)) >= half_support) {
      continue;
    }

    // Determine if any pixel falls within the antenna's kernel in this row
    // recall: pixels are arranged with a stride between them
    // for example, in thread 0, the pixel indices in the row are
    //  0, 8, 16, ... for a stride of 8.
    // So if the support is less than the stride, only one pixel
    // can fall within its kernel.

    // as long as the support is smaller than the stride, rint or roundf should
    // produce identical results

    int thread_pix_idx = rintf((antx - threadIdx.x) * inv_stride);
    thread_pix_idx = thread_pix_idx >= 0 ? thread_pix_idx : 0;
    float ant_distx = abs(antx - (thread_pix_idx * stride + threadIdx.x + 0.5));

    if (ant_distx >= half_support) {
      continue;
    }

    auto phase_ant = phases[ant];
    float scale =
        tex2D<float>(gcf_tex, abs(ant_distx * inv_half_support),
                     abs((anty - (threadIdx.y + 0.5)) * inv_half_support));

    typename FFT::value_type temp;

    __cms_f<half2>(temp.x, float2{phase_ant.x, phase_ant.y}, f_eng[ant].X,
                   scale);
    __cms_f<half2>(temp.y, float2{phase_ant.z, phase_ant.w}, f_eng[ant].Y,
                   scale);

    // re-arrange the elements in  RRII layout
    __half im = temp.x.y;
    temp.x.y = temp.y.x;
    temp.y.x = im;

    thread_data[thread_pix_idx].x += temp.x;
    thread_data[thread_pix_idx].y += temp.y;
  }
}

template <typename FFT, unsigned int Support>
__device__ void
find_valid_antennas_lwasv(const float3 *antpos,
                          unsigned long long int (&valid_ants)[4]) {

  constexpr float half_support = Support / 2.f;
  constexpr float inv_half_support = 2. / float(Support);
  constexpr int stride = float(size_of<FFT>::value) / FFT::elements_per_thread;
  constexpr double inv_stride = 1. / double(stride);
  static_assert(Support < size_of<FFT>::value / FFT::elements_per_thread,
                "Support size must be less than the stride");

  for (int ant = 0; ant < LWA_SV_NSTANDS; ++ant) {
    float antx = antpos[ant].x;
    float anty = antpos[ant].y;

    if (abs(anty - (threadIdx.y + 0.5)) >= half_support) {
      continue;
    }

    int thread_pix_idx = rintf((antx - threadIdx.x) * inv_stride);
    thread_pix_idx = thread_pix_idx >= 0 ? thread_pix_idx : 0;
    float ant_distx = abs(antx - (thread_pix_idx * stride + threadIdx.x + 0.5));

    if (ant_distx >= half_support) {
      continue;
    }

    int group = (ant) / 64;
    valid_ants[group] |= (1ull << 64 - (ant + 1 - group * 64));
  }
}

/**
 * @brief On-chip memory based gridding.
 *
 * The gridded data is written directly to the thread registers without
 * requiring any atomic operations. * For GCF kernel sizes less than the stride
 * of the FFT, each antenna can atmost contribute to one pixel (or a register).
 * The gridder loops through all antennas in each thread and write gridded
 * values to thr appropriate registers.
 *
 * @tparam FFT FFT object constructed using cuFFTDx
 * @tparam Support Support Size
 * @tparam NStands Number of antennas
 * @param thread_data Thread register array for use as temporary workspace
 * @param f_eng Pointer to the F-Engine data to be gridded
 * @param antpos Pointer to antenna position array
 * @param phases Pointer to the phases array
 * @param gcf_tex
 * @return __device__
 */
template <
    typename FFT, unsigned int Support, unsigned int NStands,
    std::enable_if_t<
        std::is_same<__half2, typename FFT::output_type::value_type>::value,
        bool> = true>
__device__ void grid_dual_pol_dx7(
    typename FFT::value_type (&thread_data)[FFT::elements_per_thread],
    const cnib2 *f_eng, const float3 *__restrict__ antpos,
    const float4 *__restrict__ phases, cudaTextureObject_t gcf_tex,
    const unsigned long long int (&valid_ants)[4]) {
  constexpr float half_support = float(Support) / 2.f;
  constexpr float inv_half_support = 2. / float(Support);
  constexpr int stride = float(size_of<FFT>::value) / FFT::elements_per_thread;
  constexpr double inv_stride = 1. / double(stride);
  static_assert(Support < size_of<FFT>::value / FFT::elements_per_thread,
                "Support size must be less than the stride");

  // 256 antennas divided into 4 groups
  for (int group = 0; group < 4; ++group) {
    auto _temp_grp = valid_ants[group];
    int offset = group * 64;
    int counter = 0;
    while (__popcll(_temp_grp) > 0) {
      ++counter;
      auto pos = __ffsll(_temp_grp);
      int ant = (offset + 63 + 1 - pos); // for 0-based indexing

      offset -= pos;
      _temp_grp >>= pos;

      if (ant < 0 || ant >= 256 || counter > 64) {
        return;
        printf("Illegal antenna %d %d %d\n", ant, threadIdx.x, threadIdx.y);
      }

      float antx = antpos[ant].x;
      float anty = antpos[ant].y;

      // Determine if any pixel falls within the antenna's kernel in this row
      // recall: pixels are arranged with a stride between them
      // for example, in thread 0, the pixel indices in the row are
      //  0, 8, 16, ... for a stride of 8.
      // So if the support is less than the stride, only one pixel
      // can fall within its kernel.

      // as long as the support is smaller than the stride, rint or roundf
      // should produce identical results

      int thread_pix_idx = rintf((antx - threadIdx.x) * inv_stride);
      thread_pix_idx = thread_pix_idx >= 0 ? thread_pix_idx : 0;
      float ant_distx =
          abs(antx - (thread_pix_idx * stride + threadIdx.x + 0.5));

      auto phase_ant = phases[ant];
      float scale = 
          tex2D<float>(gcf_tex, abs(ant_distx * inv_half_support),
                       abs((anty - (threadIdx.y + 0.5)) * inv_half_support));

      typename FFT::value_type temp;

      __cms_f<half2>(temp.x, float2{phase_ant.x, phase_ant.y}, f_eng[ant].X,
                     scale);
      __cms_f<half2>(temp.y, float2{phase_ant.z, phase_ant.w}, f_eng[ant].Y,
                     scale);

      // re-arrange the elements in  RRII layout
      __half im = temp.x.y;
      temp.x.y = temp.y.x;
      temp.y.x = im;

      thread_data[thread_pix_idx].x += temp.x;
      thread_data[thread_pix_idx].y += temp.y;
    }
  }
}