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

#ifndef SRC_EX_DATA_COPIER_CUH_
#define SRC_EX_DATA_COPIER_CUH_
#define _CG_ABI_EXPERIMENTAL

#include <cooperative_groups.h>

#include <cuda/std/type_traits>
#include <type_traits>

#include "./constants.h"
#include "./cu_helpers.cuh"
#include "./types.hpp"
namespace cg = cooperative_groups;

/**
 * @brief Copy data between two memory regions
 *
 * @tparam T Source data type
 * @tparam P Destination data type
 * @tparam C A common type to cast source and destinations if T and P are
 * different
 * @param tb Thread block
 * @param src Pointer to source memory
 * @param dst Pointer to destination memory
 * @param nelements Total elements of the type T to be copied if T and P are
 * equal otherwise number of elements of type C. This will enable. vectorized
 * copies
 * @return void
 */
template <
    typename C = void, unsigned int ThreadTileSize, typename T, typename P,
    std::enable_if_t<(ThreadTileSize & (ThreadTileSize - 1)) == 0, bool> = true>
__device__ inline void memcpy_dx(cg::thread_block tb, P* dst, const T* src,
                                 size_t nelements, unsigned int tile_rank = 0) {
  auto tile = cg::experimental::tiled_partition<ThreadTileSize>(tb);
  if (tile.meta_group_rank() != tile_rank) return;
  static_assert(
      !(std::is_same<T, P>::value == false && std::is_void<C>::value == true),
      "For different source and destination data types, a common type for the "
      "copy is required.");

  for (int i = tile.thread_rank(); i < nelements; i += tile.num_threads()) {
    if (std::is_same<P, T>::value) {
      reinterpret_cast<P*>(dst)[i] = reinterpret_cast<const P*>(src)[i];
    } else {
      reinterpret_cast<C*>(dst)[i] = reinterpret_cast<const C*>(src)[i];
    }
  }
}

/**
 * @brief Copy imaging data into the shared memory
 *
 * All the data will be arranged in a sequence: f-engine, ant pos, phases. The
 * destination memory can be of anytype. The data sequences will be
 * appropriately casted to their output types.
 *
 * @tparam T Type of the destination memory type
 * @tparam Order Data ordering of imaging data
 * @param tb Thread block
 * @param img_data ImageData struct for the data to be imaged
 * @param shared_mem Destination memory
 * @param gulp Gulp index, 0-based
 * @param chan_idx Channel index, 0-based
 * @param[out] f_eng_out Output pointer to the f-engine data
 * @param[out] ant_pos_out Output pointer to the antenna position data
 * @param[out] phases_out Output pointer to the phases data
 * @return void
 */
template <typename T, size_t ThreadTileSize, PKT_DATA_ORDER Order = CHAN_MAJOR>
__device__ void copy_lwasv_imaging_data(
    cg::thread_block tb, const uint8_t* f_eng_in, const float* antpos_in,
    const float* phases_in, T* shared_mem, size_t gulp, size_t chan_idx,
    size_t ngulps_per_seq, size_t nchan, uint8_t*& f_eng_out,
    float*& ant_pos_out, float*& phases_out, T*& f_eng_x_phases_out) {
  // copy F-engine data using the first thread group
  memcpy_dx<uint8_t, ThreadTileSize>(
      tb, shared_mem,
      GetFEngSample<Order>(f_eng_in, gulp, chan_idx, ngulps_per_seq, nchan),
      LWA_SV_NSTANDS * LWA_SV_NPOLS, 0);
  f_eng_out = reinterpret_cast<uint8_t*>(shared_mem);

  // copy antenna data using the second thread group
  auto antenna_data_start = f_eng_out + LWA_SV_NSTANDS * LWA_SV_NPOLS;
  memcpy_dx<float3, ThreadTileSize>(tb, antenna_data_start,
                                    GetAntPos(antpos_in, chan_idx),
                                    LWA_SV_NSTANDS, 1);
  ant_pos_out = reinterpret_cast<float*>(antenna_data_start);
  if (tb.thread_rank() == 0) {
  }

  // copy phases data using the third thread group
  auto phases_data_start = ant_pos_out + LWA_SV_NSTANDS * 3;
  memcpy_dx<float2, ThreadTileSize>(tb, phases_data_start,
                                    GetPhases(phases_in, chan_idx),
                                    LWA_SV_NSTANDS * LWA_SV_NPOLS, 2);
  phases_out = phases_data_start;

  f_eng_x_phases_out = reinterpret_cast<T*>(phases_out + LWA_SV_NSTANDS * 2);
  float2* phases_v = reinterpret_cast<float2*>(phases_out);
  CNib2* f_eng_v = reinterpret_cast<CNib2*>(f_eng_out);

  tb.sync();

  volatile int idx = threadIdx.y * blockDim.x + threadIdx.x;
  if (idx < LWA_SV_NSTANDS) {
    float2 grid_val_X{0, 0};
    float2 grid_val_Y{0, 0};
    __cm_f(grid_val_X, phases_v[idx], f_eng_v[idx].X);

    __cm_f(grid_val_Y, phases_v[idx + 1], f_eng_v[idx].Y);

    f_eng_x_phases_out[idx].x.x += grid_val_X.x;
    f_eng_x_phases_out[idx].x.y += grid_val_Y.x;
    f_eng_x_phases_out[idx].y.x += grid_val_X.y;
    f_eng_x_phases_out[idx].y.y += grid_val_Y.y;
  }

  if (tb.thread_rank() == 0) {
  }

  tb.sync();
}

#endif  // SRC_EX_DATA_COPIER_CUH_
