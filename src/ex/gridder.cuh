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
 * tile computes the grid elements of a single antenna at a time. A full image
 * grid is maintained in the shared memory and the grid elements are atomically
 * added to the grid. This grid only works with half precision.
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
 * @return
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

/**
 * @brief For each thread pixel group, find overlaping antenna kernels.
 *
 * The list of antennas are stored as bit positions in four 64-bit integers
 * @tparam FFT
 * @tparam Support
 * @param antpos
 * @param valid_ants
 * @return __device__
 */
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
 *
 * @note This can be potentially much more slower than the shared-memory based
 * gridder. It is possible that only a few pixel groups have a large number of
 * antennas compared to the rest, for example, at low frequencies and moderate
 * resolutions. This causes only a few threads to spend larger times gridding
 * the data thereby creating a bottleneck.
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

enum ImageDiv { UPPER, LOWER };

/**
 * @brief Grid the F-Engine data using shared-memory
 *
 * The thread block is tiled with a size of \p Support x \p Support and each
 * tile computes the grid elements of a single antenna at a time. A half-grid is
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
 * @param Div Flag to indicate the divison of the image (UPPER OR LOWER)
 * @param pix2m Multipler to convert pixels to meters
 * @return void
 *
 * @relatesalso MOFFCuHandler
 * @see grid_dual_pol_dx5
 */
template <
    typename FFT, unsigned int Support, unsigned int NStands,
    std::enable_if_t<
        std::is_same<__half2, typename FFT::output_type::value_type>::value,
        bool> = true>
__device__ inline void grid_dual_pol_dx8(
    cg::thread_block tb, const cnib2 *f_eng, const float3 *__restrict__ antpos,
    const float4 *__restrict__ phases, typename FFT::value_type *smem,
    cudaTextureObject_t gcf_tex, ImageDiv Div = UPPER, float pix2m=1.0) {
  using complex_type = typename FFT::value_type;
  constexpr float half_support = Support / 2.f;
  constexpr float inv_support = 1.f / float(Support);
  constexpr float inv_half_support = 1.f / float(half_support);
  auto tile = cg::tiled_partition<Support * Support>(tb);
  constexpr float im_ymid = size_of<FFT>::value / 2 - 0.5;
  constexpr int half_grid = size_of<FFT>::value / 2;

  /* static_assert(
      size_of<FFT>::value == 128,
      "grid_dual_pol_dx8 (half-gridder) is only specialized for 128 sq. pix"); */
  // if(blockIdx.x==2 && threadIdx.x==0 && threadIdx.y==0){
  //     printf("pix2m: %f\n", pix2m);
  //   }

  for (int ant = tile.meta_group_rank(); ant < NStands;
       ant += tile.meta_group_size()) {
    typename FFT::value_type temp_data; //__half2half2(0);
    temp_data.x = __half2half2(0);
    temp_data.y = __half2half2(0);
    float antx = antpos[ant].x;
    float anty = antpos[ant].y;

    bool is_ant_out = (Div == UPPER ? ((anty - half_support) > im_ymid)
                                    : ((anty + half_support) < (im_ymid+1)));

    if (is_ant_out) {
      continue;
    }

    // calculate the position of the cell on the UV plane this thread will write
    // to
    float v = int(tile.thread_rank() * inv_support) - (half_support) + 0.5;
    float u = tile.thread_rank() -
              int(tile.thread_rank() * inv_support) * Support - (half_support) +
              0.5;

    // check if the cell falls within the current half of the image
    bool is_cell_valid = true;
    is_cell_valid &=
        (0 <= int(antx + u)) && (int(antx + u) < size_of<FFT>::value);
    is_cell_valid &=
        (Div == UPPER ? (0 <= int(anty + v)) && (int(anty + v) < half_grid)
                      : (half_grid <= int(anty + v)) &&
                            (int(anty + v) < size_of<FFT>::value));
    is_cell_valid &= (abs(int(antx + u) + 0.5 - antx) < half_support &&
                      abs(int(anty + v) + 0.5 - anty) < half_support);

    // only calculate the grid value if it's valid
    // This conditional causes thread divergence, which can be removed by
    // setting the scale to zero for out of bound pixels. However, another
    // conditional must be introduced for the atomic add.

    
    if (is_cell_valid) {
      float scale = tex2D<float>(
          gcf_tex, abs((int(antx + u) + 0.5 - antx) * pix2m),
          abs((int(anty + v) + 0.5 - anty) * pix2m));

      // dirty beam calculation
      // cnib _t;
      // _t.re=1;
      // _t.im=1;
      // auto phase_ant = phases[ant];
      // __cms_f(temp_data.x, float2{1,1}, _t,
      //         scale);
      // __cms_f(temp_data.y, float2{1,1}, _t,
      //         scale);

      auto phase_ant = phases[ant];
      __cms_f(temp_data.x, float2{phase_ant.x, phase_ant.y}, f_eng[ant].X,
              scale);
      __cms_f(temp_data.y, float2{phase_ant.z, phase_ant.w}, f_eng[ant].Y,
              scale);

      // Re-arrange into RRII layout
      __half im = temp_data.x.y;
      temp_data.x.y = temp_data.y.x;
      temp_data.y.x = im;

      int offset = Div == UPPER ? 0 : size_of<FFT>::value / 2;
      if ((int(antx + u) + (int(anty + v) - offset) * size_of<FFT>::value) >
          (size_of<FFT>::value * size_of<FFT>::value) / 2) {
        printf("Invalid antenna: %f %f %f %f %f %f %d\n", antx, anty, u, v,
               antx + u, anty + v, size_of<FFT>::value);
      }
      atomicAdd(
          &smem[int(antx + u) + (int(anty + v) - offset) * size_of<FFT>::value]
               .x,
          temp_data.x);
      atomicAdd(
          &smem[int(antx + u) + (int(anty + v) - offset) * size_of<FFT>::value]
               .y,
          temp_data.y);
    }
    tile.sync();
  }
}


template <
    typename FFT, unsigned int Support, unsigned int NStands,
    std::enable_if_t<
        std::is_same<__half2, typename FFT::output_type::value_type>::value,
        bool> = true>
__device__ inline void grid_dual_pol_dx9(
    cg::thread_block tb, const cnib2 *f_eng, const float3 *__restrict__ antpos,
    const float4 *__restrict__ phases, typename FFT::value_type *smem,
    float* gcf_grid_elem, ImageDiv Div = UPPER) {
      // auto tb = cg::this_thread_block();
      constexpr int nelements = Support * Support;
      constexpr float inv_nelements = 1.f/float(nelements);

      // divide threads into groups of nelements. Each group grids one antenna at a time
      int this_thread_ant = tb.thread_rank() * inv_nelements;
      int this_thread_elem = tb.thread_rank() - this_thread_ant * nelements;
      constexpr int half_support = Support/2;
      constexpr int half_grid = size_of<FFT>::value/2;
      constexpr int nants_per_pass = (FFT::block_dim.x * FFT::block_dim.y)/nelements;
      int offset = Div == UPPER ? 0 : size_of<FFT>::value / 2;

      if(this_thread_ant >= nants_per_pass){//cannot grid with these threads
        return;
      }
      int channel_idx = blockIdx.x;

      for(int ant=this_thread_ant;ant<NStands;ant+=nants_per_pass){

        typename FFT::value_type temp_data; //__half2half2(0);
        temp_data.x = __half2half2(0);
        temp_data.y = __half2half2(0);


        int dy = half_support - (this_thread_elem) / (Support);
        int dx = (this_thread_elem) % (Support) - half_support;

        float antx = antpos[ant].x;
        float anty = antpos[ant].y;

        int xpix = int(antx + dx);
        int ypix = int(anty + dy);

        if((Div==UPPER && ypix>=half_grid)|| (Div==LOWER && ypix<half_grid)){
          continue;
        }
        if(xpix<0 || xpix>=size_of<FFT>::value){
          continue;
        }

        float scale = gcf_grid_elem[channel_idx * NStands * nelements + ant * nelements + this_thread_elem];
        auto phase_ant = phases[ant];
        // cnib _t;
        // _t.re=1;
        // _t.im=1;
        //For some reason, there is a 1-pixel offset between epic and wsclean
        // images. For now, simply shift the epic sky by 1-pixel in the +y direction
        float c = cosf(3.14/64. * ypix);
        float s = sinf(3.14/64. * ypix); 
        __cms_f(temp_data.x, float2{phase_ant.x*c-phase_ant.y*s, phase_ant.y*c+phase_ant.x*s},f_eng[ant].X,
                scale);
        __cms_f(temp_data.y, float2{phase_ant.z, phase_ant.w}, f_eng[ant].Y,
                scale);

        // Re-arrange into RRII layout
        __half im = temp_data.x.y;
        temp_data.x.y = temp_data.y.x;
        temp_data.y.x = im;

        // if(blockIdx.x==0 && (xpix>=64 || (ypix-offset)>=64 || xpix<0 || ypix<0) || (xpix + (ypix-offset) * size_of<FFT>::value)>=2048){
        //   printf("xpix ypix antx anty dx dy: %d %d %f %f %d %d\n",xpix, ypix,antx,anty,dx,dy);
        // }

        atomicAdd(&smem[xpix + (ypix-offset) * size_of<FFT>::value].x, temp_data.x);
        atomicAdd(&smem[xpix + (ypix-offset) * size_of<FFT>::value].y, temp_data.y);
      }


      // constexpr offset = (FFT::block_dim.x * FFT::block_dim.y) % nelements == 0? 0:1;
};