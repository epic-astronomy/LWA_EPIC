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

#include "./channel_descriptor.h"
#include "./constants.h"
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
  void destroy_textures(cudaArray_t& p_tex_arr, cudaTextureObject_t& p_tex_obj);

  /// @brief Reset GCF texture object
  /// @param p_gcf_tex_dim Size of the GCF texture
  /// @param p_gcf_2D_ptr Host pointer to the GCF kernel array
  void reset_gcf_tex(int p_gcf_tex_dim, float* p_gcf_2D_ptr);

  /// @brief Reset antenna positions on device
  /// @param p_nchan Number of channels
  /// @param p_antpos_ptr Host pointer to the antenna position array
  void reset_antpos(int p_nchan, float* p_antpos_ptr);

  /// @brief Reset phases on device
  /// @param p_nchan Number of channels
  /// @param p_phases_ptr Host pointer to the phases array
  void reset_phases(int p_nchan, float* p_phases_ptr);

  /// Number of streams to split a gulp into
  int m_nstreams;
  std::unique_ptr<cudaStream_t[]> m_gulp_custreams{nullptr};
  void create_gulp_custreams();

  /// GCF kernel elements for each antenna and frequency
  float* m_gcf_elem;
  bool is_m_gcf_elem_set{false};

  /// GCF correction kernel
  /// The inverse of the iFFT of this kernel must be multipled with the final
  /// image The correction kernel can be the GCF itself or an average of the
  /// GCFs of individual antennas
  float* m_correction_kernel_d{nullptr};
  bool is_correction_kernel_set{false};
  float* m_correction_grid_d{nullptr};
  bool is_correction_grid_set{false};
  void get_correction_kernel(float* p_out_kernel, int p_support_size,
                             int p_nchan);
  void set_correction_grid(float* p_corr_grid, int p_grid_size, int p_nchan);

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
  void set_imaging_kernel();
  void set_img_grid_dim();

  bool use_bf16_accum{false};

 public:
  MOFFCuHandler(){}
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
  void reset_data(int p_nchan, size_t p_nseq_per_gulp, float* p_ant_pos,
                  float* p_phases);

  /**
   * @brief Allocate device memory to store F-Engine data
   *
   * @param nbytes Byte to allocate
   */
  void allocate_f_eng_gpu(size_t nbytes);

  /**
   * @brief Image a gulp of data
   *
   * @param[in] p_data_ptr Host pointer to the F-Engine data
   * @param[out] p_out_ptr Device pointer to store the output data
   * @param p_first Flag if the gulp is the first one in the accumulation
   * @param p_last Flag if the gulp is the last one in the accumulation
   */
  void process_gulp(uint8_t* p_data_ptr, float* p_out_ptr = nullptr,
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
  void process_gulp_old(uint8_t* p_data_ptr, size_t p_buf_size,
                        float* p_out_ptr, size_t p_out_size,
                        bool p_first = true, bool p_last = false);

  /**
   * @brief Allocate device memory to store output image
   *
   * @param nbytes Bytes to allocate
   */
  void allocate_out_img(size_t nbytes);

  void reset_gcf_elem(int p_nchan, int p_support, int p_chan0, float p_delta,
                      int p_grid_size);

  ~MOFFCuHandler();
};

#endif  // SRC_EX_MOFF_CU_HANDLER_H_
