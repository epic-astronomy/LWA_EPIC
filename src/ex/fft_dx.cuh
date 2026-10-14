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
 * @param gcf_grid_elem_cplx Optional complex (w-projected) kernel elements,
 * interleaved re/im with the same layout as gcf_grid_elem (half2 values for
 * WMode == kGcfGlobalHalf). nullptr selects the real kernel path.
 * @param acorr_kernels Optional per-stand auto-correlations of the kernel
 * weights from ComputeAcorrKernels, [chan][stand][(2 support - 1)^2]. When
 * set, the antenna auto-correlations are removed from the image at the end of
 * the gulp; nullptr leaves them in.
 * @param seq_begin, seq_end Sequences of the gulp imaged by this launch
 * (seq_end < 0: to the end). A gulp can be imaged by several launches on
 * consecutive chunks, each adding its sum into the output; this keeps the
 * 16-bit on-chip accumulator from stalling (MOFFCuHandler m_seq_chunk).
 * @param acorr_power_g With acorr_kernels and chunked launches: per-channel
 * stand powers [chan][stand] carried between the chunks of a gulp, so that
 * only the gulp's last chunk removes the auto-correlations (one FFT per gulp).
 * nullptr: every launch removes its own.
 * @tparam WMode Where the gridder keeps the kernel weights (GcfWeightMode)
 */
/**
 * @brief Whether a channel's kernel weights (4 B per cell) fit in the shared
 * memory left after the grid, antenna positions and auto-correlations, given
 * the 1.5 x FFT::shared_memory_size allocation made by SetImagingKernel.
 */
template <class FFT, int support>
__host__ __device__ constexpr bool GcfSharedFits() {
  return LWA_SV_NSTANDS * support * support * 4 <=
         int(FFT::shared_memory_size / 2) -
             int(LWA_SV_NSTANDS * (sizeof(float3) + sizeof(float4)));
}

template <
    typename FFT, int support = 2, typename accum_oc_t = __half2,
    PKT_DATA_ORDER Order = CHAN_MAJOR, int WMode = kGcfGlobal,
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
                          float* gcf_correction_grid = nullptr,
                          float* gcf_grid_elem_cplx = nullptr,
                          const float2* __restrict__ acorr_kernels = nullptr,
                          int seq_begin = 0, int seq_end = -1,
                          float4* __restrict__ acorr_power_g = nullptr) {
  using complex_type = typename FFT::value_type;
  extern __shared__ complex_type shared_mem[];

  constexpr int kPasses = GcfAntPasses<FFT, support, LWA_SV_NSTANDS>();
  constexpr int kWPasses =
      (WMode == kGcfReg || WMode == kGcfRegHalf) ? kPasses : 1;
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

  // Kernel weights for the voltage gridder. They are fixed for the launch, so
  // the staged modes load this channel's weights once here instead of once per
  // sequence (see GcfWeightMode).
  constexpr int nelements = support * support;
  GcfWeights<WMode, kWPasses> weights;
  weights.real = gcf_grid_elem;
  weights.cplx = gcf_grid_elem_cplx;
  if constexpr (WMode == kGcfShared) {
    static_assert(GcfSharedFits<FFT, support>(),
                  "Kernel weights do not fit in the spare shared memory");
    void* staged = auto_corrs + LWA_SV_NSTANDS;
    const size_t chan_base = size_t(channel_idx) * LWA_SV_NSTANDS * nelements;
    for (int i = tb.thread_rank(); i < LWA_SV_NSTANDS * nelements; i += tb.size()) {
      if (gcf_grid_elem_cplx != nullptr) {
        float2 v = reinterpret_cast<const float2*>(gcf_grid_elem_cplx)[chan_base + i];
        reinterpret_cast<__half2*>(staged)[i] = __floats2half2_rn(v.x, v.y);
      } else {
        reinterpret_cast<float*>(staged)[i] = gcf_grid_elem[chan_base + i];
      }
    }
    weights.staged = staged;
  }
  if constexpr (WMode == kGcfReg || WMode == kGcfRegHalf) {
    constexpr int nants_per_pass =
        (FFT::block_dim.x * FFT::block_dim.y) / nelements;
    // Same (float) thread-to-cell mapping as DualPolEpicHalfGridder
    constexpr float inv_nelements = 1.f / float(nelements);
    const int my_ant = tb.thread_rank() * inv_nelements;
    const int my_elem = tb.thread_rank() - my_ant * nelements;
#pragma unroll
    for (int k = 0; k < kPasses; ++k) {
      const int ant = my_ant + k * nants_per_pass;
      float2 v{0.f, 0.f};
      if (my_ant < nants_per_pass && ant < LWA_SV_NSTANDS) {
        const size_t idx =
            (size_t(channel_idx) * LWA_SV_NSTANDS + ant) * nelements + my_elem;
        if (gcf_grid_elem_cplx != nullptr) {
          v = reinterpret_cast<const float2*>(gcf_grid_elem_cplx)[idx];
        } else {
          v.x = gcf_grid_elem[idx];
        }
      }
      if constexpr (WMode == kGcfReg) {
        weights.reg[k] = v;
      } else {
        weights.reg_h[k] = __floats2half2_rn(v.x, v.y);
      }
    }
  }
  tb.sync();

  // loop over each sequence in the gulp for channel_idx channel
  if (seq_end < 0) {
    seq_end = nseq_per_gulp;
  }
  for (int seq_no = seq_begin; seq_no < seq_end; ++seq_no) {
    // volatile int idx = 200;
    for (int i = tb.thread_rank();
         i < size_of<FFT>::value * size_of<FFT>::value / 2; i += tb.size()) {
      shared_mem[i] = __half2half2(0);
    }
    for (int _reg = 0; _reg < FFT::elements_per_thread; ++_reg) {
      thread_data[_reg] = __half2half2(0);
    }
    // The grid must be cleared before any thread grids into it
    tb.sync();

    DualPolEpicFullGridder<FFT, support, LWA_SV_NSTANDS, CNib2, WMode, kWPasses>(
        tb,
        reinterpret_cast<const CNib2*>(GetFEngSample<Order>(
            f_eng_g, seq_no, channel_idx, nseq_per_gulp, nchan)),
        // reinterpret_cast<const float3 *>(GetAntPos(antpos_g, channel_idx)),
        antpos_smem,
        reinterpret_cast<const float4*>(GetPhases(phases_g, channel_idx)),
        shared_mem,
        thread_data,
        weights, NONE, channel_idx);

    tb.sync();

    // Accumulate auto corrs
    if (acorr_kernels != nullptr) {
      AcorrAccumulatorF(tb,
        reinterpret_cast<const CNib2*>(GetFEngSample<Order>(
                f_eng_g, seq_no, channel_idx, nseq_per_gulp, nchan)),
        auto_corrs);
    }


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

  // Remove the antenna auto-correlations (Thyagarajan et al. 2017, sec. 3.2).
  // Stand a contributes P_a g_a g_a^* |K^_a(l, m)|^2 to every image, where P_a
  // is its power accumulated over the gulp. That is the transform of
  // P_a g_a g_a^* A_a gridded around the uv origin, with A_a the
  // auto-correlation of its own weights (ComputeAcorrKernels), so one more
  // FFT gives the whole term for all stands.
  // With chunked launches the powers are carried in acorr_power_g and the
  // term is removed once, by the gulp's last chunk.
  bool acorr_now = acorr_kernels != nullptr;
  if (acorr_now && acorr_power_g != nullptr && !(seq_begin == 0 && seq_end >= nseq_per_gulp)) {
    float4* carried = acorr_power_g + size_t(channel_idx) * LWA_SV_NSTANDS;
    const bool last_chunk = seq_end >= nseq_per_gulp;
    for (int ant = tb.thread_rank(); ant < LWA_SV_NSTANDS; ant += tb.size()) {
      if (last_chunk) {
        auto_corrs[ant] = auto_corrs[ant] + carried[ant];
      } else {
        carried[ant] = seq_begin == 0 ? auto_corrs[ant] : carried[ant] + auto_corrs[ant];
      }
    }
    acorr_now = last_chunk;
    tb.sync();
  }
  if (acorr_now) {
    constexpr int s2 = 2 * support - 1;
    constexpr int ncell = s2 * s2;
    // Fft2D normalises by 1/64 in TransposeTri; the image accumulates
    // |FFT / 64|^2 * row_size per sequence.
    constexpr float kFftNorm = 64.f;
    const auto* acorr_phases =
        reinterpret_cast<const float4*>(GetPhases(phases_g, channel_idx));
    const float2* akern =
        acorr_kernels + size_t(channel_idx) * LWA_SV_NSTANDS * ncell;
    // Float scratch in the grid memory, which is free between FFTs:
    // [0] total power, then per cell (XX + i YY) and XY* coefficients (re, im).
    float* scratch = reinterpret_cast<float*>(shared_mem);
    tb.sync();
    for (int i = tb.thread_rank(); i < 1 + 4 * ncell; i += tb.size()) {
      scratch[i] = 0.f;
    }
    tb.sync();
    for (int ant = tb.thread_rank(); ant < LWA_SV_NSTANDS; ant += tb.size()) {
      const float4 g = acorr_phases[ant];
      atomicAdd(&scratch[0], auto_corrs[ant].x * (g.x * g.x + g.y * g.y) +
                                 auto_corrs[ant].y * (g.z * g.z + g.w * g.w));
    }
    tb.sync();
    // Normalise by the total power so the half-precision FFT input stays
    // near unity whatever the data and gain scale.
    const float total = scratch[0];
    if (total > 0.f) {
      const float inv = 1.f / total;
      for (int i = tb.thread_rank(); i < LWA_SV_NSTANDS * ncell; i += tb.size()) {
        const float2 a = akern[i];
        if (a.x == 0.f && a.y == 0.f) {
          continue;
        }
        const int ant = i / ncell;
        const int cell = i - ant * ncell;
        const float4 g = acorr_phases[ant];
        const float4 p = auto_corrs[ant];
        const float wx = p.x * (g.x * g.x + g.y * g.y) * inv;
        const float wy = p.y * (g.z * g.z + g.w * g.w) * inv;
        // (XY*) gx gy^*
        const float gr = g.x * g.z + g.y * g.w, gi = g.y * g.z - g.x * g.w;
        const float cr = (p.z * gr - p.w * gi) * inv;
        const float ci = (p.z * gi + p.w * gr) * inv;
        float* c = scratch + 1 + 4 * cell;
        atomicAdd(c + 0, wx * a.x - wy * a.y);
        atomicAdd(c + 1, wx * a.y + wy * a.x);
        atomicAdd(c + 2, cr * a.x - ci * a.y);
        atomicAdd(c + 3, cr * a.y + ci * a.x);
      }
      tb.sync();
      // This thread holds grid row threadIdx.y, columns threadIdx.x + reg *
      // stride (DualPolEpicFullGridder). Cell offsets wrap around the origin.
      const int row = threadIdx.y;
      const int drow = row < support ? row : row - row_size;
      for (int _reg = 0; _reg < FFT::elements_per_thread; ++_reg) {
        const int col = threadIdx.x + _reg * stride;
        const int dcol = col < support ? col : col - row_size;
        thread_data[_reg] = __half2half2(0);
        if (drow > -support && drow < support && dcol > -support && dcol < support) {
          const float* c = scratch + 1 + 4 * ((drow + support - 1) * s2 + dcol + support - 1);
          // RRII: x = (Re slot 1, Re slot 2), y = (Im slot 1, Im slot 2)
          thread_data[_reg].x = __floats2half2_rn(c[0], c[2]);
          thread_data[_reg].y = __floats2half2_rn(c[1], c[3]);
        }
      }
      tb.sync();
      Fft2D<FFT>(tb, thread_data, shared_mem, false /*fftshift*/);
      const float k = float(row_size) * total / kFftNorm;
      for (int _reg = 0; _reg < FFT::elements_per_thread; ++_reg) {
        // slot 1 -> XX (real part) and YY (imaginary part); slot 2 -> XY*
        const float xx = __half2float(thread_data[_reg].x.x) * k;
        const float yy = __half2float(thread_data[_reg].y.x) * k;
        const float xy_re = __half2float(thread_data[_reg].x.y) * k;
        const float xy_im = __half2float(thread_data[_reg].y.y) * k;
        stokes_xx_yy[_reg] += accum_oc_t{_accum_t(-xx), _accum_t(-yy)};
        // ComputeUU / ComputeVV are Re / Im (XY*)
        stokes_U_V[_reg] += accum_oc_t{_accum_t(-xy_re), _accum_t(-xy_im)};
      }
    }
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
  // Kernel weights are staged in shared memory once per launch where they fit
  // in the spare shared memory (128x128 grid, support <= 5; production uses
  // 5). Otherwise they are read from global memory every sequence. On an RTX 4090 at
  // production settings this is ~4 ms per 40 ms gulp faster than kGcfGlobal
  // (tests/gpu, gulp_budget --weight-mode).
  switch (support) {
    case 1:
      return (void*)(BlockEpicImager<FFT, 1, accum_oc_t, CHAN_MAJOR,
                                     GcfSharedFits<FFT, 1>() ? kGcfShared : kGcfGlobal>);
    case 2:
      return (void*)(BlockEpicImager<FFT, 2, accum_oc_t, CHAN_MAJOR,
                                     GcfSharedFits<FFT, 2>() ? kGcfShared : kGcfGlobal>);
    case 3:
      return (void*)(BlockEpicImager<FFT, 3, accum_oc_t, CHAN_MAJOR,
                                     GcfSharedFits<FFT, 3>() ? kGcfShared : kGcfGlobal>);
    case 4:
      return (void*)(BlockEpicImager<FFT, 4, accum_oc_t, CHAN_MAJOR,
                                     GcfSharedFits<FFT, 4>() ? kGcfShared : kGcfGlobal>);
    case 5:
      return (void*)(BlockEpicImager<FFT, 5, accum_oc_t, CHAN_MAJOR,
                                     GcfSharedFits<FFT, 5>() ? kGcfShared : kGcfGlobal>);
    case 6:
      return (void*)(BlockEpicImager<FFT, 6, accum_oc_t, CHAN_MAJOR,
                                     GcfSharedFits<FFT, 6>() ? kGcfShared : kGcfGlobal>);
    case 7:
      return (void*)(BlockEpicImager<FFT, 7, accum_oc_t, CHAN_MAJOR,
                                     GcfSharedFits<FFT, 7>() ? kGcfShared : kGcfGlobal>);
    case 8:
      return (void*)(BlockEpicImager<FFT, 8, accum_oc_t, CHAN_MAJOR,
                                     GcfSharedFits<FFT, 8>() ? kGcfShared : kGcfGlobal>);
    case 9:
      return (void*)(BlockEpicImager<FFT, 9, accum_oc_t, CHAN_MAJOR,
                                     GcfSharedFits<FFT, 9>() ? kGcfShared : kGcfGlobal>);
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
