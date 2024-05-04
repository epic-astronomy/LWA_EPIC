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

#ifndef SRC_EX_FFT_DX_CUH_
#define SRC_EX_FFT_DX_CUH_
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cufftdx.hpp>
#include <type_traits>

#include "./constants.h"
#include "./cu_helpers.cuh"
#include "./gridder.cuh"
#include "./types.hpp"

using namespace cufftdx;
namespace cg = cooperative_groups;

/**
 * @brief Perform FFT shift along the X-dimension
 * 
 * @tparam FFT 
 * @param thread_data Thread registers that store the fft elements
 */
template<class FFT>
void __device__ FftShift(typename FFT::value_type (&thread_data)[FFT::elements_per_thread]){
  constexpr auto nswaps = FFT::elements_per_thread/2;
  for(int i=0;i<nswaps;++i){
      // this is equivalent to a circular shift
      // remember: the data are stored with a "stride".
      // that means we can simply swap elements within
      // each thread to perform an fftshift
      auto temp_swap = thread_data[i];
      thread_data[i] = thread_data[i+nswaps];
      thread_data[i+nswaps] = temp_swap;
  }
}

/**
 * @brief Perform 2D FFT
 * 
 * @tparam FFT FFT object constructed using cuFFTDx
 * @param tb thread block
 * @param smem pointer to the workspace shared memory
 * @param fftshift flag to indicate whether to fftshift
 */
template<class FFT>
void __device__ Fft2D(cg::thread_block tb, typename FFT::value_type (&thread_data)[FFT::elements_per_thread], typename FFT::value_type* smem, bool fftshift){
  constexpr auto row_size = size_of<FFT>::value;
  constexpr int stride = row_size / FFT::elements_per_thread;
  FFT().execute(thread_data, smem /*, workspace*/);
  if(fftshift){
    FftShift<FFT>(thread_data);
  }
  tb.sync();
  ///////////////////////////////////////
  // To complete the IFFT, transpose the grid and IFFT on it, which is
  //  equivalent to a column-wise IFFT.
  // Load everything into shared memory and normalize.
  // This ensures there is no overflow.
  TransposeTri<FFT>(thread_data, smem,
                     /*_norm=*/half(1.) / half(64.));
  FFT().execute(thread_data, smem /*, workspace*/);
  if(fftshift){
    FftShift<FFT>(thread_data);
  }
  tb.sync();
}

/**
 * @brief Block fft specialization for half precision
 *
 * For half-precision, the data will be stored as a complex<half2>
 * with an RRII (real real img img) layout.
 * This can be exploited to speedup the computation.
 * Here grid elements from two grids are coalesced and represented as a single
 * grid That means each complex type holds two half-precision complex numbers
 * from two images at the same grid point, respectively, and cuFFTDx takes care
 * of doing FFT/IFFT on individual grids. This enables storing data for both the
 * polarizations at a single place allowing fast computation of XX, YY, XY*, X*Y
 * products.
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
 *                  The grid size will be automatically deduced from the FFT
 * object.
 * @param chan_offset Offset to the first channel in the current stream.
 * @param is_first_gulp Flag if the passed data belongs to the first gulp. It
 * determines whether to assign or increment the output image block.
 * @param chan0 Channel number of the first channel
 * @param lmbda_scale Multiplier to convert wavelength from meters to
 * meters/pixel aka sampling length
 * @param gcf_grid_elem Individual kernel elements for each antenna/frequency
 * @param gcf_correction_grid Pixel-based correction factors for each frequency
 */
template <
    typename FFT, int support = 2, typename accum_oc_t = __half2,
    PKT_DATA_ORDER Order = CHAN_MAJOR,
    std::enable_if_t<
        std::is_same<__half2, typename FFT::output_type::value_type>::value,
        bool> = true>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void BlockEpicImager(const uint8_t* f_eng_g,
                          const float* __restrict__ antpos_g,
                          const float* __restrict__ phases_g, int nseq_per_gulp,
                          int nchan, cudaTextureObject_t gcf_tex,
                          float* output_g, int chan_offset = 0,
                          bool is_first_gulp = true,
                          float* gcf_grid_elem = nullptr,
                          float* gcf_correction_grid = nullptr) {
  using complex_type = typename FFT::value_type;
  extern __shared__ complex_type shared_mem[];

  auto VoltageGridderF = DualPolEpicFullGridder<FFT, support, LWA_SV_NSTANDS>;
  auto AutoCorrGridderF = DualPolEpicFullGridder<FFT, support, LWA_SV_NSTANDS,float4>;
  auto AcorrAccumulatorF = AccumAutoCorrs<FFT, LWA_SV_NSTANDS>;

  // constexpr int fft_steps = 2;  // row-wise followed by column-wise
  constexpr int stride = size_of<FFT>::value / FFT::elements_per_thread;
  constexpr int row_size =
      size_of<FFT>::value;  // blockDim.x * FFT::elements_per_thread;

  complex_type thread_data[FFT::elements_per_thread];

  using accum_g_t = float4;  // precision for global memory accumulator
  using _accum_t = decltype(accum_oc_t::x);

  accum_oc_t stokes_xx_yy[FFT::elements_per_thread] = {accum_oc_t{0., 0.}};
  accum_oc_t stokes_U_V[FFT::elements_per_thread] = {accum_oc_t{0., 0.}};
  

  auto tb = cg::this_thread_block();

  // Storage mode: nchan, x, y, npol
  // This produces one write instruction per pixel for 4 cross-pols
  accum_g_t* out_polv = reinterpret_cast<accum_g_t*>(output_g);

  int channel_idx = blockIdx.x + chan_offset;

  //   copy antenna positions into shared memory
  float3* antpos_smem = reinterpret_cast<float3*>(
      shared_mem + FFT::shared_memory_size / sizeof(complex_type));

  // reserve memory to store auto corrs
  // the storage layout is a follows
  // ac.x.x = Re(X*X)
  // ac.x.y = Re(Y*Y)
  // ac.y.x = Re(XY*)
  // ac.y.y = Im(XY*)
  float4* auto_corrs = reinterpret_cast<float4*>(antpos_smem + LWA_SV_NSTANDS);
  for (int ant = tb.thread_rank(); ant < LWA_SV_NSTANDS; ant += tb.size()) {
    //printf("%d %d\n",ant, tb.thread_rank());
    auto_corrs[ant] = {0};
  }

  auto* _antpos_g =
      reinterpret_cast<const float3*>(GetAntPos(antpos_g, channel_idx));

  for (int i = tb.thread_rank(); i < LWA_SV_NSTANDS; i += tb.size()) {
    antpos_smem[i] = _antpos_g[i];
  }
  tb.sync();

  // loop over each sequence in the gulp for channel_idx channel
  for (int seq_no = 0; seq_no < nseq_per_gulp; ++seq_no) {
    // volatile int idx = 200;
    for (int i = tb.thread_rank();
         i < size_of<FFT>::value * size_of<FFT>::value / 2; i += tb.size()) {
      shared_mem[i] = __half2half2(0);
    }
    for (int _reg = 0; _reg < FFT::elements_per_thread; ++_reg) {
      thread_data[_reg] = __half2half2(0);
    }

    VoltageGridderF(tb,
        reinterpret_cast<const CNib2*>(GetFEngSample<Order>(
            f_eng_g, seq_no, channel_idx, nseq_per_gulp, nchan)),
        // reinterpret_cast<const float3 *>(GetAntPos(antpos_g, channel_idx)),
        antpos_smem,
        reinterpret_cast<const float4*>(GetPhases(phases_g, channel_idx)),
        shared_mem,
        thread_data,
        gcf_grid_elem, NONE);

    tb.sync();

    // Accumulate auto corrs
    AcorrAccumulatorF(tb,
      reinterpret_cast<const CNib2*>(GetFEngSample<Order>(
              f_eng_g, seq_no, channel_idx, nseq_per_gulp, nchan)),
      auto_corrs);


    Fft2D<FFT>(tb, thread_data, shared_mem,false /*fftshift*/);

    // Accumulate cross-pols using on-chip memory.
    // This would lead to spilling and may result in reduced occupancy
    // Using f16 instead of f32 can alleviate this issue
    for (int _reg = 0; _reg < FFT::elements_per_thread; ++_reg) {
      
      _accum_t xx = ComputeXX<_accum_t, FFT>(thread_data[_reg]);
      _accum_t yy = ComputeYY<_accum_t, FFT>(thread_data[_reg]);
      _accum_t uu = ComputeUU<_accum_t, FFT>(thread_data[_reg]);
      _accum_t vv = ComputeVV<_accum_t, FFT>(thread_data[_reg]);

      // stokes[_reg]+=accum_g_t{xx, yy};
      _accum_t mult=_accum_t(row_size);
      // scale the pixel values to reduce errors
      stokes_xx_yy[_reg] += accum_oc_t{xx*mult, yy*mult};
      stokes_U_V[_reg] += accum_oc_t{uu*mult, vv*mult};
    }

    
    tb.sync();
  }  // End of gulp loop
 // reset registers and shared mem
  for (int i = tb.thread_rank();
         i < size_of<FFT>::value * size_of<FFT>::value / 2; i += tb.size()) {
      shared_mem[i] = __half2half2(0);
    }
    for (int _reg = 0; _reg < FFT::elements_per_thread; ++_reg) {
      thread_data[_reg] = __half2half2(0);
    }
  // calculate the XX* and YY* auto-corrs in two independent FFTs
  AutoCorrGridderF(tb,auto_corrs, antpos_smem,
        reinterpret_cast<const float4*>(GetPhases(phases_g, channel_idx)),
        shared_mem,
        thread_data,
        gcf_grid_elem, XX_YY);
  tb.sync();
  Fft2D<FFT>(tb, thread_data, shared_mem,true);
  tb.sync();
  for (int _reg = 0; _reg < FFT::elements_per_thread; ++_reg) {
    // remove XX* and YY* autocorrs
    stokes_xx_yy[_reg] += accum_oc_t{-_accum_t(__habs(thread_data[_reg].x.x)) * _accum_t(row_size*row_size), -_accum_t(__habs(thread_data[_reg].x.y))*_accum_t(row_size*row_size)};
    // stokes_xx_yy[_reg] -= thread_data[_reg].x.y;
  }

  // Write the final accumulated image into global memory
  for (int _reg = 0; _reg < FFT::elements_per_thread; ++_reg) {
    auto index = (threadIdx.x + _reg * stride) + threadIdx.y * row_size;
    auto gcf_corr =
        gcf_correction_grid[channel_idx * row_size * row_size + index];
    auto stk_corrected = make_v4_s<accum_oc_t, accum_g_t>(
        stokes_xx_yy[_reg], stokes_U_V[_reg], gcf_corr);
    out_polv[channel_idx * row_size * row_size + index] =
        is_first_gulp ? stk_corrected
                      : out_polv[channel_idx * row_size * row_size + index] +
                            stk_corrected;
  }
}

using FFT64x64 =
    decltype(Size<64>() + Precision<half>() + Type<fft_type::c2c>() +
             Direction<fft_direction::inverse>() + SM<890>() +
             ElementsPerThread<4>() + FFTsPerBlock<128>() + Block());

using FFT128x128 =
    decltype(Size<128>() + Precision<half>() + Type<fft_type::c2c>() +
             Direction<fft_direction::inverse>() + SM<890>() +
             ElementsPerThread<16>() + FFTsPerBlock<256>() + Block());

using FFT81x81 =
    decltype(Size<81>() + Precision<half>() + Type<fft_type::c2c>() +
             Direction<fft_direction::inverse>() + SM<890>() +
             ElementsPerThread<9>() + FFTsPerBlock<162>() + Block());

using FFT100x100 =
    decltype(Size<100>() + Precision<half>() + Type<fft_type::c2c>() +
             Direction<fft_direction::inverse>() + SM<890>() +
             ElementsPerThread<10>() + FFTsPerBlock<200>() + Block());

/**
 * @brief Get the imaging kernel object for a given support and image size
 *
 * @tparam FFT cuFFTDx template for the FFT
 * @param support Size of the support. Cannot be larger than
 * MAX_ALLOWED_SUPPORT_SIZE
 * @return void* Pointer to the imaging template instance
 */
template <class FFT, typename accum_oc_t>
void* GetImagingKernel(int support = 3) {
  switch (support) {
    case 1:
      return (void*)(BlockEpicImager<FFT, 1, accum_oc_t>);
    case 2:
      return (void*)(BlockEpicImager<FFT, 2, accum_oc_t>);
    case 3:
      return (void*)(BlockEpicImager<FFT, 3, accum_oc_t>);
    case 4:
      return (void*)(BlockEpicImager<FFT, 4, accum_oc_t>);
    case 5:
      return (void*)(BlockEpicImager<FFT, 5, accum_oc_t>);
    case 6:
      return (void*)(BlockEpicImager<FFT, 6, accum_oc_t>);
    case 7:
      return (void*)(BlockEpicImager<FFT, 7, accum_oc_t>);
    case 8:
      return (void*)(BlockEpicImager<FFT, 8, accum_oc_t>);
    case 9:
      return (void*)(BlockEpicImager<FFT, 9, accum_oc_t>);
    default:
      assert(false && "Unsupported support size");
      return 0;
  }
};

template void* GetImagingKernel<FFT128x128, float2>(int support);

// template void*
// GetImagingKernel<FFT128x128, __nv_bfloat162>(int support);

template void* GetImagingKernel<FFT64x64, float2>(int support);

// template void*
// GetImagingKernel<FFT64x64, __nv_bfloat162>(int support);
#endif  // SRC_EX_FFT_DX_CUH_
