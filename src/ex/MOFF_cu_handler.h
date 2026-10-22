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

#ifndef SRC_EX_MOFF_CU_HANDLER_H_
#define SRC_EX_MOFF_CU_HANDLER_H_

#include <stdint.h>

#include <memory>
#include <vector>

#include "./channel_descriptor.h"
#include "./constants.h"
#include "./fx_imager.h"
#include "./metrics.hpp"
#include "./rfi_stage.h"
#include "./tcc_tap.h"
#include "./types.hpp"

/**
 * @brief Handler for the GPU-side operations of `MOFFCorrelator`
 *
 */
class MOFFCuHandler {
 protected:
  /// Channel format for the GCF. Defaults to float
  cudaChannelFormatDesc m_gcf_chan_desc{
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat)};

  /// Flag if the gcf texture object is set on the GPU
  bool is_gcf_tex_set{false};
  /// Flag if the antenna position array is set on the GPU
  bool is_antpos_set{false};
  /// Flag if the phases array is set on the GPU
  bool is_phases_set{false};
  /// Flag if memory for F-Engine data is allocated on the GPU
  bool is_f_eng_cu_allocated{false};
  /// Flag if the output image memory is set on the GPU
  bool is_out_mem_set{false};
  /// Device pointer to the antenna positions data
  float* m_antpos_cu;
  /// Device pointer to the phases data
  float* m_phases_cu;
  /// Device pointer to the output image data
  float* m_output_cu;
  /// Device pointer to the F-Engine memory block
  uint8_t* m_f_eng_cu;
  /// Byte-size of the voltage data gulp
  int m_f_eng_bytes{0};
  /// Byte-size of the output image
  int m_out_img_bytes{0};

  /// CUDA texture array to store the GCF kernel
  cudaArray_t m_gcf_tex_arr;
  /// CUDA texture object for the GCF
  cudaTextureObject_t m_gcf_tex{0};
  /// GCF texture's resource description
  cudaResourceDesc m_gcf_res_desc;
  /** GCF texture's description object.
   * The address mode is set to clamp, texture access to
  normalized float, filter mode to linear, and read mode to element
  */
  cudaTextureDesc m_gcf_tex_desc;

  /// @brief Destroy all the texture objects
  /// @param p_tex_arr Texture array object
  /// @param p_tex_obj Texture object
  void DestroyTextures(cudaArray_t& p_tex_arr, cudaTextureObject_t& p_tex_obj);

  /// @brief Reset GCF texture object
  /// @param p_gcf_tex_dim Size of the GCF texture
  /// @param p_gcf_2D_ptr Host pointer to the GCF kernel array
  void ResetGcfTex(int p_gcf_tex_dim, float* p_gcf_2D_ptr);

  /// @brief Reset antenna positions on device
  /// @param p_nchan Number of channels
  /// @param p_antpos_ptr Host pointer to the antenna position array
  void ResetAntpos(int p_nchan, float* p_antpos_ptr);

  /// @brief Reset phases on device
  /// @param p_nchan Number of channels
  /// @param p_phases_ptr Host pointer to the phases array
  void ResetPhases(int p_nchan, float* p_phases_ptr);

  /// @brief Overwrite the device phases in place (same size as the last
  /// ResetPhases), e.g. with calibration gains folded in. Synchronous; call
  /// only between gulps (ProcessGulp returns after all streams finish).
  void UpdatePhases(int p_nchan, const float* p_phases_ptr);

  /// GPU calibration tap (null: off). Correlates the gulps from m_f_eng_cu.
  std::unique_ptr<TccTap> m_tcc_tap{nullptr};
  // The validity masks of the last gulps (packet_processor.hpp), in a ring of
  // mapped host buffers: the slots of lost packets are zeroed on the GPU after
  // each stream's copy (the receiver leaves them, see ZeroLostPackets)
  static constexpr int kMaskSlots = 8;
  uint8_t* m_mask_ring_h{nullptr};
  uint8_t* m_mask_ring_d{nullptr};
  size_t m_mask_slot_bytes{0};
  size_t m_mask_words{0};
  int m_mask_slot{0};
  bool m_have_mask{false};
  void ZeroLostPackets(int p_chan_offset, cudaStream_t p_stream);
  // the host gulp is time-major [seq][chan][input] (the NDP receiver's
  // layout): staged on the device and transposed to [chan][seq][input]
  bool m_time_major{false};
  uint8_t* m_tmaj_cu{nullptr};
  /// Copy stream p_i's channels of the host gulp to the channel-major device gulp and zero its lost packets
  void UploadStream(int p_i, const uint8_t* p_host, cudaStream_t p_stream);
  /// FX imaging (--imager fx): replaces the E-field kernel when set
  std::unique_ptr<FxImager> m_fx{nullptr};
  int m_fx_size{0};
  double m_image_unix{0};
  /// Recorded on each gulp stream once its part of the gulp is on the GPU
  std::vector<cudaEvent_t> m_stream_ready;

  /// Interference projection and impulse detection before imaging (null:
  /// off). When active the imaging kernel reads its output buffer instead of
  /// m_f_eng_cu, which the calibration tap keeps reading raw.
  std::unique_ptr<RfiStage> m_rfi{nullptr};
  /// (stream, sequence) pairs zeroed by the impulse detector in the last image
  uint64_t m_last_impulse_flagged{0};

  /// Number of streams to split a gulp into
  int m_nstreams;
  std::unique_ptr<cudaStream_t[]> m_gulp_custreams{nullptr};
  void CreateGulpCuStreams();

  /// GCF kernel elements for each antenna and frequency
  float* m_gcf_elem;
  bool is_m_gcf_elem_set{false};

  /// Optional complex (w-projected) kernel elements, interleaved re/im with the
  /// same layout as m_gcf_elem. nullptr selects the real kernel path.
  float* m_gcf_elem_cplx{nullptr};

  /// Per-stand auto-correlations of the gridding weights (ComputeAcorrKernels),
  /// [chan][stand][(2 support - 1)^2]. nullptr leaves the antenna
  /// auto-correlations in the image.
  float2* m_acorr_kernels{nullptr};
  /// Stand powers carried between the chunked launches of a gulp
  /// [chan][stand], so the auto-correlations are removed once per gulp
  float4* m_acorr_power{nullptr};
  /// Sequences per kernel launch (0 = the whole gulp in one launch)
  int m_seq_chunk{0};
  bool m_remove_autocorrs{false};
  void UpdateAcorrKernels(int p_nchan, int p_support);

  /// GCF correction kernel
  /// The inverse of the iFFT of this kernel must be multipled with the final
  /// image The correction kernel can be the GCF itself or an average of the
  /// GCFs of individual antennas
  float* m_correction_kernel_d{nullptr};
  bool is_correction_kernel_set{false};
  float* m_correction_grid_d{nullptr};
  bool is_correction_grid_set{false};
  void GetCorrectionKernel(float* p_out_kernel, int p_support_size,
                           int p_nchan);
  void SetCorrectionGrid(float* p_corr_grid, int p_grid_size, int p_nchan);

  /// Total number of channels per sequence
  int m_nchan_in{0};
  /// Number of sequences per gulp
  int m_nseq_per_gulp;
  OutImgDesc m_out_img_desc;
  /// Channels to be processed per stream
  int m_nchan_per_stream{0};
  /// Byte-size of voltage data to be copied per stream
  int m_nbytes_f_eng_per_stream{0};
  /// Byte-size of output image data to be copied per stream
  int m_nbytes_out_img_per_stream{0};

  int m_device_id{0};

  /// Pointer to the imaging kernel
  void* m_imaging_kernel{nullptr};
  dim3 m_img_grid_dim;
  dim3 m_img_block_dim;
  int m_shared_mem_size;
  int m_support_size;
  void SetImagingKernel();
  void SetImgGridDim();

  bool use_bf16_accum{false};

  unsigned int m_gulp_exec_gauge_id{0};

 public:
  MOFFCuHandler() {}
  // __host__ void test();
  /**
   * @brief Reset antpos, phases, GCF data on device
   *
   * @param p_nchan Total number channels in each sequence
   * @param p_nseq_per_gulp Number of sequences per gulp
   * @param p_gcf_dim Dimensions of the GCF texture
   * @param p_ant_pos Host pointer to the antenna positions array
   * @param p_phases Host pointer to the phases array
   */
  void ResetData(int p_nchan, size_t p_nseq_per_gulp, float* p_ant_pos,
                 float* p_phases);

  /**
   * @brief Allocate device memory to store F-Engine data
   *
   * @param nbytes Byte to allocate
   */
  void AllocateFEngGpu(size_t nbytes);

  /**
   * @brief Image a gulp of data
   *
   * @param[in] p_data_ptr Host pointer to the F-Engine data
   * @param[out] p_out_ptr Device pointer to store the output data
   * @param p_first Flag if the gulp is the first one in the accumulation
   * @param p_last Flag if the gulp is the last one in the accumulation
   */
  void ProcessGulp(uint8_t* p_data_ptr, float* p_out_ptr = nullptr,
                   bool p_first = true, bool p_last = false, int p_chan0 = 0,
                   float p_delta = 1.0);

  /**
   * @brief Image a gulp of data
   *
   * @param p_data_ptr Host pointer to the F-Engine data
   * @param p_buf_size Size of the buffer
   * @param p_out_ptr Device pointer to store the output data
   * @param p_out_size Size of the output in bytes
   * @param p_first Flag if the gulp is the first one in the accumulation
   * @param p_last Flag if the gulp is the last one in the accumulation
   * @param p_chan0 Channel number for the first channel in the gulp
   * @param p_delta Scaling length to convert wavelength in meters to
   * meters/pixel
   */
  [[deprecated("Use ProcessGulp instead")]] void ProcessGulpOld(
      uint8_t* p_data_ptr, size_t p_buf_size, float* p_out_ptr,
      size_t p_out_size, bool p_first = true, bool p_last = false);

  /**
   * @brief Allocate device memory to store output image
   *
   * @param nbytes Bytes to allocate
   */
  void AllocateOutImg(size_t nbytes);

  void ResetGcfElem(int p_nchan, int p_support, int p_chan0, float p_delta,
                    int p_grid_size);

  /**
   * @brief Copy the real GCF kernel elements to the host
   *
   * @param p_out Host buffer of p_nchan * LWA_SV_NSTANDS * p_support^2 floats
   */
  void GetGcfElem(float* p_out, int p_nchan, int p_support);

  /**
   * @brief Upload complex (w-projected) kernel elements and switch the gridder
   * to them. Passing nullptr frees them and restores the real kernel path.
   *
   * @param p_in Host buffer of p_nchan * LWA_SV_NSTANDS * p_support^2 complex
   * values, interleaved re/im
   */
  void SetGcfElemCplx(const float* p_in, int p_nchan, int p_support);

  /**
   * @brief Remove the antenna auto-correlations from the images. Takes effect
   * when the gridding weights are next set (SetGcfElemCplx).
   */
  void SetRemoveAutocorrs(bool p_remove);

  /**
   * @brief Image each gulp in kernel launches of p_seq_chunk sequences (0 =
   * one launch). Each launch adds its sum into the output image, so the
   * 16-bit on-chip accumulator never sums more than p_seq_chunk terms; summed
   * over a whole 1000-sequence gulp it stalls and the image comes out ~24%
   * low. Each extra launch costs ~0.17 ms per gulp on an RTX 4090.
   */
  void SetSeqChunk(int p_seq_chunk);

  /// Start the GPU calibration tap (tensor-core correlator, tcc_tap.h) on the
  /// first p_nchan channels of gulps of p_nseq sequences.
  void EnableTccTap(const TccTapOptions& p_opts, int p_nchan, int p_nseq);
  /// Image with the FX imager at p_size x p_size instead of the E-field kernel
  void EnableFx(int p_size) { m_fx_size = p_size; }
  bool UsesFx() const { return m_fx_size > 0; }
  /// FX: regrid every image to the all-sky monitor's pixels as well
  void SetSkyTable(const double* p_xy_a, const double* p_xy_b, int p_capacity, double p_t_a, double p_t_b,
                   int p_nbin) {
    if (m_fx) m_fx->SetSkyTable(p_xy_a, p_xy_b, p_capacity, p_t_a, p_t_b, p_nbin);
  }
  /// Which samples of the next gulp arrived, for the FX imager and the GPU
  /// calibration tap (FxImager::SetGulpValidity)
  void SetGulpValidity(const uint64_t* p_mask, size_t p_words, const uint8_t* p_block);
  /// FX: peel the bright compact sources (FxImager::SetPeel)
  void SetPeel(const double* p_frame, double p_dl, double p_umin_lambda, bool p_sky_only = false) {
    if (m_fx) m_fx->SetPeel(p_frame, p_dl, p_umin_lambda, p_sky_only);
  }
  /// Layout of the next gulps on the host: time-major (t_maj) or channel-major
  void SetGulpTimeMajor(bool p_time_major) { m_time_major = p_time_major; }
  int SkyPixels() const { return m_fx ? m_fx->SkyPixels() : 0; }
  bool SkyDone() const { return m_fx && m_fx->SkyDone(); }
  /// FX: start time (unix) of the image whose last gulp comes next, and
  /// where in its host buffer the pixel spectra go (floats after the start)
  void SetImageTime(double p_unix) { m_image_unix = p_unix; }
  size_t SkyPixelOffset() const { return size_t(m_nchan_in) * m_fx_size * m_fx_size * 4; }
  uint32_t SkyPeelMask() const { return m_fx ? m_fx->SkyPeelMask() : 0u; }
  bool SkyPeelTerm() const { return m_fx && m_fx->SkyPeelTerm(); }
  /// Wait until the last FX image is in its host buffer
  void FinishFxImage() {
    if (m_fx) m_fx->Sync();
  }
  /// (Re)create the FX imager for the current channels and hand it the
  /// geometry (stand cells, kernel weights, gridding correction)
  void SetupFx(int p_nchan, int p_nseq);
 private:
  void ProcessGulpFx(uint8_t* p_data_ptr, uint8_t* p_img_in, bool p_rfi, float* p_out_ptr, bool p_first,
                     bool p_last);
 public:

  /// Enable the interference stage (projector uploads via SetProjector;
  /// impulse detection when p_impulse_sigma > 0). Call before the first gulp.
  void EnableRfiStage(float p_impulse_sigma);
  /// Projector components in use (0: none)
  int GetProjQ() const { return m_rfi ? m_rfi->Q() : 0; }
  /// (stream, sequence) pairs zeroed by impulse detection in the last image
  uint64_t GetImpulseFlagged() const { return m_last_impulse_flagged; }

  /// Describe the next gulp to the GPU calibration tap; call before
  /// ProcessGulp. Returns true if the gulp will be correlated.
  bool ArmTccTap(uint64_t p_seq_start, int64_t p_chan0, int p_nchan, int p_nseq, float p_quality) {
    return m_tcc_tap && m_tcc_tap->Arm(p_seq_start, p_chan0, p_nchan, p_nseq, p_quality);
  }

  ~MOFFCuHandler();
};

#endif  // SRC_EX_MOFF_CU_HANDLER_H_
