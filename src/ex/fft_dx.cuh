#ifndef FFTDX_CUH
#define FFTDX_CUH
#include "constants.h"
//#include "data_copier.cuh"
#include "gridder.cuh"
#include "types.hpp"
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cufftdx.hpp>
#include <type_traits>

using namespace cufftdx;
namespace cg = cooperative_groups;

__global__ void
_temp(float* ant_pos, cudaTextureObject_t gcf)
{
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("inside\n");
        printf("ant pos[0]: %f %f %f\n", ant_pos[0], ant_pos[1], ant_pos[2]);
    }
}

/**
 * @brief Block fft specialization for half precision
 *
 * For half-precision, the data will be stored as a complex<half2>
 * with an RRII (real real img img) layout.
 * This can be exploited to speedup the computation.
 * Here grid elements from two grids are coalesced and represented as a single grid
 * That means each complex type holds two half-precision complex numbers from two images
 * at the same grid point, respectively, and cuFFTDx takes care of doing FFT/IFFT
 * on individual grids. This enables storing data for both the polarizations at a single place
 * allowing fast computation of XX, YY, XY*, X*Y products.
 *
 * @tparam FFT FFT type constructed using cufftdx
 * @tparam Order Packet data ordering type
 * @param f_eng_g Pointer to the F-Engine data in global memory
 * @param antpos_g Pointer to the antenna positions in global memory
 * @param phases_g Pointer to the phases array in global memory
 * @param nseq_per_gulp Number of sequences in each gulp
 * @param nchan Number of channels in each sequence
 * @param gcf_tex GCF texture object
 * @param output_g[out] Pointer to the output image. Must have dimensions of
 *                  \p nseq_per_gulp, \p nchan, \p grid_size, \p grid_size, 2.
 *                  The grid size will be automatically deduced from the FFT object.
 * @param chan_offset Offset to the first channel in the current stream.
 * @param is_first_gulp Flag if the passed data belongs to the first gulp. It determines whether to assign or increment the output image block.
 */
template<typename FFT, PKT_DATA_ORDER Order = CHAN_MAJOR, std::enable_if_t<std::is_same<__half2, typename FFT::output_type::value_type>::value, bool> = true>
__launch_bounds__(FFT::max_threads_per_block)
  __global__ void block_fft_kernel(
    const uint8_t* f_eng_g,
    const float* antpos_g,
    const float* phases_g,
    size_t nseq_per_gulp,
    size_t nchan,
    cudaTextureObject_t gcf_tex,
    float* output_g,
    size_t chan_offset = 0,
    bool is_first_gulp=true)
{

    using complex_type = typename FFT::value_type;
    extern __shared__ complex_type shared_mem[];

    complex_type thread_data[FFT::storage_size];

    for (int seq_no = 0; seq_no < nseq_per_gulp; ++seq_no) {
        volatile int idx = 0;
        volatile int channel_idx = blockIdx.x + chan_offset;

        constexpr int stride = size_of<FFT>::value / FFT::elements_per_thread;
        constexpr int row_size = size_of<FFT>::value; // blockDim.x * FFT::elements_per_thread;
        for (int _reg = 0; _reg < FFT::elements_per_thread; ++_reg) {
            shared_mem[(threadIdx.x + _reg * stride) * row_size + threadIdx.y] = __half2half2(0);
        }

        grid_dual_pol_dx5<FFT, 2, LWA_SV_NSTANDS>(
          cg::this_thread_block(),
          thread_data,
          reinterpret_cast<const cnib2*>(
            get_f_eng_sample<Order>(
              f_eng_g, seq_no, channel_idx, nseq_per_gulp, nchan)),
          reinterpret_cast<const float3*>(get_ant_pos(antpos_g, channel_idx)),
          reinterpret_cast<const float4*>(get_phases(phases_g, channel_idx)),
          shared_mem,
          gcf_tex);

        __syncthreads();
        for (int _reg = 0; _reg < FFT::elements_per_thread; ++_reg) {
            auto index = (threadIdx.x + _reg * stride) * row_size + threadIdx.y;
            thread_data[_reg] = shared_mem[index];
        }

        // if (threadIdx.x == 0 && threadIdx.y == 0 && channel_idx == 0 && i == idx)
        //     printf("thread: %f\n", __half2float(thread_data[idx].x.x));

        // Execute IFFT (row-wise)
        FFT().execute(thread_data, shared_mem /*, workspace*/);
        ///////////////////////////////////////
        // To complete the IFFT, transpose the grid and IFFT on it, which is
        //  equivalent to a column-wise IFFT.
        // Load everything into shared memory and normalize
        // this ensures there is no overflow
        half _norm = half(1) / half(row_size);
        // #pragma unroll
        for (int _reg = 0; _reg < FFT::elements_per_thread; ++_reg) {
            auto index = (threadIdx.x + _reg * stride) * row_size + threadIdx.y;
            shared_mem[index].x.x = thread_data[_reg].x.x * _norm;
            shared_mem[index].x.y = thread_data[_reg].x.y * _norm;
            shared_mem[index].y.x = thread_data[_reg].y.x * _norm;
            shared_mem[index].y.y = thread_data[_reg].y.y * _norm;
        }

        __syncthreads();

        // transpose the matrix
        for (int _reg = 0; _reg < FFT::elements_per_thread; ++_reg) {
            auto index = (threadIdx.x + _reg * stride) + threadIdx.y * row_size;
            thread_data[_reg] = shared_mem[index];
        }
        __syncthreads();
        // execute column-wise iFFT
        FFT().execute(thread_data, shared_mem);

        for (int _reg = 0; _reg < FFT::elements_per_thread; ++_reg) {
            auto index = (threadIdx.x + _reg * stride) + threadIdx.y * row_size;
            auto xx_yy = thread_data[_reg].x * thread_data[_reg].x + thread_data[_reg].y * thread_data[_reg].y;
            if (seq_no == 0) {
                output_g[channel_idx * row_size * row_size + index] = float(xx_yy.x + xx_yy.y);
            } else {
                output_g[channel_idx * row_size * row_size + index] += float(xx_yy.x + xx_yy.y);
            }
        }

        if (channel_idx == 0 && seq_no == idx) {
            // for (int i = 0; i < FFT::elements_per_thread; ++i) {
            //     printf("thread: %f %f %d %d %d\n",__half2float(thread_data[i].x.x), __half2float(thread_data[i].y.x), threadIdx.x, threadIdx.y, i);
            // }
            // printf("thread: %f\n", __half2float(thread_data[idx].x.x));
        }
    }
}

using FFT64x64 = decltype(Size<64>() + Precision<half>() + Type<fft_type::c2c>() + Direction<fft_direction::inverse>() + SM<860>() + ElementsPerThread<8>() + FFTsPerBlock<128>() + Block());
#endif // FFTDX_CUH