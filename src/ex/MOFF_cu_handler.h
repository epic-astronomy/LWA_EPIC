#ifndef MOFF_CU_HANDLER_H
#define MOFF_CU_HANDLER_H
#include "channel_descriptor.h"

#include <stdint.h>

/**
 * @brief Handler for the GPU-side operations of `MOFFCorrelator`
 * 
 */
class MOFFCuHandler
{
  protected:
    /// Channel format for the GCF. Defaults to float
    cudaChannelFormatDesc m_gcf_chan_desc{
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat)
    };

    /// Flag if the gcf texture object is set on the GPU
    bool is_gcf_tex_set{ false };
    /// Flag if the antenna position array is set on the GPU
    bool is_antpos_set{ false };
    /// Flag if the phases array is set on the GPU
    bool is_phases_set{ false };
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

    /// CUDA texture array to store the GCF kernel
    cudaArray_t m_gcf_tex_arr;
    /// CUDA texture object for the GCF
    cudaTextureObject_t m_gcf_tex{ 0 };
    /// GCF texture's resource description
    cudaResourceDesc m_gcf_res_desc;
    /** GCF texture's description object. The address mode is set to clamp, texture access to 
    normalized float, filter mode to linear, and read mode to element */
    cudaTextureDesc m_gcf_tex_desc;

    /// @brief Destroy all the texture objects
    /// @param p_tex_arr Texture array object
    /// @param p_tex_obj Texture object
    void destroy_textures(cudaArray_t& p_tex_arr, cudaTextureObject_t& p_tex_obj);

    /// @brief Reset GCF texture object
    /// @param p_support Support size
    /// @param p_gcf_2D_ptr Host pointer to the GCF kernel array
    void reset_gcf_tex(int p_support, float* p_gcf_2D_ptr);

    /// @brief Reset antenna positions on device
    /// @param p_nchan Number of channels
    /// @param p_antpos_ptr Host pointer to the antenna position array
    void reset_antpos(int p_nchan, float* p_antpos_ptr);

    /// @brief Reset phases on device
    /// @param p_nchan Number of channels
    /// @param p_phases_ptr Host pointer to the phases array
    void reset_phases(int p_nchan, float* p_phases_ptr);

    /// Total number of channels per sequence
    int m_nchan;
    /// Number of sequences 
    int m_nseq_per_gulp;

  public:
    MOFFCuHandler(){};
    // __host__ void test();
    /**
     * @brief Reset antpos, phases, GCF data on device
     * 
     * @param p_nchan Total number channels in each sequence
     * @param p_nseq_per_gulp Number of sequences per gulp
     * @param p_gcf_dim Dimensions of the GCF texture
     * @param p_ant_pos Host pointer to the antenna positions array
     * @param p_phases Host pointer to the phases array
     * @param p_gcf_2D Host pointer to the GCF 2D kernel array
     */
    void reset_data(int p_nchan,size_t p_nseq_per_gulp, int p_gcf_dim, float* p_ant_pos, float* p_phases, float* p_gcf_2D = nullptr);

    /**
     * @brief Allocate device memory to store F-Engine data
     * 
     * @param nbytes Byte to allocate
     */
    void allocate_f_eng_gpu(size_t nbytes);

    /**
     * @brief Image a gulp of data
     * 
     * @param p_data_ptr Host pointer to the F-Engine data
     * @param p_buf_size Size of the buffer
     * @param p_out_ptr Device pointer to store the output data
     * @param p_out_size Size of the output in bytes
     */
    void process_gulp(uint8_t* p_data_ptr, size_t p_buf_size, float* p_out_ptr, size_t p_out_size);

    /**
     * @brief Allocate device memory to store output image
     * 
     * @param nbytes Bytes to allocate
     */
    void allocate_out_img(size_t nbytes);
    ~MOFFCuHandler();
};

#endif // MOFF_CU_HANDLER_H


// cudaChannelFormatDesc m_antpos_chan_desc
// {
//     cudaCreateChannelDesc(32, 32, 32, 0, cudaChannelFormatKindFloat)
// };
// cudaChannelFormatDesc m_phase_chan_desc
// {
//     cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat)
// };

// bool is_host_gcf_texture_set{ false };
// bool is_antpos_tex_set{ false };
// bool is_phases_tex_set{ false };
// cudaArray_t m_antpos_tex_arr;
// cudaTextureObject_t m_antpos_tex{0};
// cudaResourceDesc m_antpos_res_desc;
// cudaTextureDesc m_antpos_tex_desc;

// cudaArray_t m_phases_tex_arr;
// cudaTextureObject_t m_phases_tex{0};
// cudaResourceDesc m_phases_res_desc;
// cudaTextureDesc m_phases_tex_desc;

// void reset_antpos_tex(int p_nchan, float* p_antpos_ptr);
// void reset_phases_tex(int p_nchan, float* p_phases);

// void reset_1D_texture(bool p_reallocate, cudaArray_t& p_tex_arr, cudaTextureObject_t& p_tex_obj, cudaChannelFormatDesc& p_chan_desc, cudaResourceDesc& p_res_desc, cudaTextureDesc& p_tex_desc, int p_width, int p_depth, float* p_host_ptr);
// void set_default_tex_res_desc(cudaResourceDesc& p_res_desc, cudaTextureDesc& p_tex_desc, cudaArray_t& p_tex_arr);
