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
                  //   typename FFT::value_type* f_eng_x_phases,
                  typename FFT::value_type* smem,
                  cudaTextureObject_t gcf_tex)
{
    using complex_type = typename FFT::value_type;
    constexpr float half_support = Support / 2.f;
    constexpr float inv_support = 1.f / float(Support);
    constexpr float inv_half_support = 1.f / float(half_support);
    // constexpr int stride = size_of<FFT>::value / FFT::elements_per_thread;
    // int block_height = 32 / blockDim.x;
    // auto tb = this_thread_block();
    auto tile = cg::experimental::tiled_partition<Support * Support>(tb);
    // int width = Support * Support > warpSize ? warpSize : Support * Support;

    // if (cg::this_thread_block().thread_rank() >= Support * Support) {
    //     return;
    // }
    // #pragma unroll
    for (int ant = tile.meta_group_rank(); ant < NStands; ant += tile.meta_group_size()) {
        float scale = 1;
        // for (int ant = 0; ant < NStands; ++ant) {
        thread_data[0] = __half2half2(0);
        // thread_data[1] = thread_data[0];
        float antx = antpos[ant].x;
        float anty = antpos[ant].y;

        // auto grid_val = f_eng_x_phases[ant];
        // int row = (tile.thread_rank() / Support);
        // int col = tile.thread_rank() - row * Support;

        float v = int(tile.thread_rank() * inv_support) - (half_support) + 0.5;
        float u = tile.thread_rank() - int(tile.thread_rank() * inv_support) * Support - (half_support) + 0.5;

        // if(threadIdx.x==0 && blockIdx.x==0 && threadIdx.y==0){
        //     printf("u: %f v:%f ant: %f %f uant: %f vant: %f\n", u, v, antx, anty,abs(int(antx + u) + 0.5 - antx),  abs(int(anty + v) + 0.5 - anty));
        // }

        // float incx = u < 0 ? -0.5 : 0.5;
        // float incy = v < 0 ? -0.5 : 0.5;

        // float dx = abs(u + incx - antpos[ant].x);
        // float dy = abs(v + incy - antpos[ant].y);

        // u = u < 0 ? u : u ;
        // v = v < 0 ? v : v ;

        if (!((0 <= int(antx + u) <= size_of<FFT>::value) &&
              (0 <= int(anty + v) <= size_of<FFT>::value) &&
              abs(int(antx + u) + 0.5 - antx) < half_support &&
              abs(int(anty + v) + 0.5 - anty) < half_support)) {
            // continue;
            // __syncwarp();
            // scale = 0;
        } else {
            auto phase_ant = phases[ant];
            // scale = tex2D<float>(
            //   gcf_tex,
            //   abs(int((int(antx + u) + 0.5 - antx) * inv_half_support * 32)),
            //   abs(int((int(anty + v) + 0.5 - anty) * inv_half_support * 32)));

            scale = tex2D<float>(
              gcf_tex,
              abs((int(antx + u) + 0.5 - antx) * inv_half_support),
              abs((int(anty + v) + 0.5 - anty) * inv_half_support));

            __cms_f(thread_data[0].x, float2{ phase_ant.x, phase_ant.y }, f_eng[ant].X, scale);
            __cms_f(thread_data[0].y, float2{ phase_ant.z, phase_ant.w }, f_eng[ant].Y, scale);

            // thread_data[1].x.x = thread_data[0].x.x;
            // thread_data[1].x.y = thread_data[0].y.x;
            // thread_data[1].y.x = thread_data[0].x.y;
            // thread_data[1].y.y = thread_data[0].y.y;

            thread_data[1] = thread_data[0];
            thread_data[0].x.y = thread_data[1].y.x;
            thread_data[0].y.x = thread_data[1].x.y;

            // if (threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0) {
            //     printf("u: %f v:%f uant: %f vant: %f scale: %f\n", u, v, abs(int(antx + u) + 0.5 - antx), abs(int(anty + v) + 0.5 - anty),scale);
            // }

            // half2 temp = __half2half2(half(ant * scale));

            // smem[int(antx + u) + int(anty + v) * size_of<FFT>::value].x += thread_data[1].x;
            // smem[int(antx + u) + int(anty + v) * size_of<FFT>::value].y += thread_data[1].y;
            atomicAdd(&smem[int(antx + u) + int(anty + v) * size_of<FFT>::value].x, thread_data[0].x);
            atomicAdd(&smem[int(antx + u) + int(anty + v) * size_of<FFT>::value].y, thread_data[0].y);
            // __syncwarp();
        }

        // if (abs(int(antx + u + 0.5) + incx - antx) < half_support &&
        //     abs(int(anty + v + 0.5) + incy - anty) < half_support) {
        //     continue;
        //     scale = 0;
        // }
    }
}

// int xpos = threadIdx.x + i * stride;
// int ypos = threadIdx.y;

// float minx = __reduce_min_sync(WARP_MASK, threadIdx.x + i * stride) + 0.5;
// float maxx = __reduce_max_sync(WARP_MASK, threadIdx.x + i * stride) + 0.5;
// float miny = __reduce_min_sync(WARP_MASK, threadIdx.y) + 0.5;
// float maxy = __reduce_max_sync(WARP_MASK, threadIdx.y) + 0.5;

// if(tile32.meta_group_rank()==0 && blockIdx.x==0 && tile32.thread_rank()==0 && ant==0){
//         printf("%f %f %f %f:\n",minx,maxx,miny,maxy);
//     }

// __half2 grid_X;
// __half2 grid_Y;

// if (tile32.thread_rank() == 0) {
// antx =
// anty =
// }
// antx = __shfl_sync(WARP_MASK, antx,0);
// anty = __shfl_sync(WARP_MASK, anty, 0);

// if (tile32.thread_rank() == 0) {
// antx = antpos[ant].x;
// anty = antpos[ant].y;