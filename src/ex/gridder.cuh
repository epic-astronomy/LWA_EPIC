#pragma once

#include "constants.h"
#include "cu_helpers.cuh"
#include "types.hpp"
#include <cooperative_groups.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <cufftdx.hpp>

namespace cg = cooperative_groups;
using namespace cufftdx;

#define WARP_MASK 0xFFFFFFFF

/**
 * @brief Grid the F-Engine data
 *
 * The thread block is tiled with a size of \p Support x \p Support and each tile
 * computes the grid elements of a single antenna at a time. A grid is maintained in
 * the shared memory and the grid elements are atomically added to the grid.
 * This grid only works with half precision.
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
template<typename FFT, unsigned int Support, unsigned int NStands, std::enable_if_t<std::is_same<__half2, typename FFT::output_type::value_type>::value, bool> = true>
__device__ inline void
grid_dual_pol_dx5(cg::thread_block tb,
                  typename FFT::value_type thread_data[FFT::elements_per_thread],
                  const cnib2* f_eng,
                  const float3* __restrict__ antpos,
                  const float4* __restrict__ phases,
                  typename FFT::value_type* smem,
                  cudaTextureObject_t gcf_tex)
{
    using complex_type = typename FFT::value_type;
    constexpr float half_support = Support / 2.f;
    constexpr float inv_support = 1.f / float(Support);
    constexpr float inv_half_support = 1.f / float(half_support);
    auto tile = cg::experimental::tiled_partition<Support * Support>(tb);
   
    for (int ant = tile.meta_group_rank(); ant < NStands; ant += tile.meta_group_size()) {
        float scale = 1;
        thread_data[0] = __half2half2(0);
        float antx = antpos[ant].x;
        float anty = antpos[ant].y;

        float v = int(tile.thread_rank() * inv_support) - (half_support) + 0.5;
        float u = tile.thread_rank() - int(tile.thread_rank() * inv_support) * Support - (half_support) + 0.5;

        if (!((0 <= int(antx + u) <= size_of<FFT>::value) &&
              (0 <= int(anty + v) <= size_of<FFT>::value) &&
              abs(int(antx + u) + 0.5 - antx) < half_support &&
              abs(int(anty + v) + 0.5 - anty) < half_support)) {
        } else {
            auto phase_ant = phases[ant];
            scale = tex2D<float>(
              gcf_tex,
              abs((int(antx + u) + 0.5 - antx) * inv_half_support),
              abs((int(anty + v) + 0.5 - anty) * inv_half_support));

            __cms_f(thread_data[0].x, float2{ phase_ant.x, phase_ant.y }, f_eng[ant].X, scale);
            __cms_f(thread_data[0].y, float2{ phase_ant.z, phase_ant.w }, f_eng[ant].Y, scale);

            thread_data[1] = thread_data[0];
            thread_data[0].x.y = thread_data[1].y.x;
            thread_data[0].y.x = thread_data[1].x.y;

            atomicAdd(&smem[int(antx + u) + int(anty + v) * size_of<FFT>::value].x, thread_data[0].x);
            atomicAdd(&smem[int(antx + u) + int(anty + v) * size_of<FFT>::value].y, thread_data[0].y);
        }

    }
}