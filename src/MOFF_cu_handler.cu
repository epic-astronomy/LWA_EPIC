#include "ex/MOFF_cu_handler.h"
#include "ex/constants.h"
#include "ex/cu_helpers.cuh"
#include "ex/fft_dx.cuh"
#include "ex/gridder.cuh"
#include "ex/types.hpp"
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <iostream>

namespace cg = cooperative_groups;

void
MOFFCuHandler::reset_antpos(int p_nchan, float* p_antpos_ptr)
{
    cudaSetDevice(m_device_id);
    if (is_antpos_set) {
        cudaFree(m_antpos_cu);
    }
    auto nbytes = LWA_SV_NSTANDS * p_nchan * 3 * sizeof(float);
    cudaMalloc(&m_antpos_cu, nbytes);
    cudaMemcpy(m_antpos_cu, p_antpos_ptr, nbytes, cudaMemcpyHostToDevice);
    is_antpos_set = true;
}

void
MOFFCuHandler::reset_phases(int p_nchan, float* p_phases_ptr)
{
    cudaSetDevice(m_device_id);
    if (is_phases_set) {
        cudaFree(m_phases_cu);
    }

    auto nbytes = LWA_SV_NSTANDS * p_nchan * LWA_SV_NPOLS * 2 * sizeof(float);
    cudaMalloc(&m_phases_cu, nbytes);
    cudaMemcpy(m_phases_cu, p_phases_ptr, nbytes, cudaMemcpyHostToDevice);

    is_phases_set = true;
}

void
MOFFCuHandler::reset_gcf_tex(int p_gcf_tex_dim, float* p_gcf_2D_ptr)
{
    cudaSetDevice(m_device_id);
    if (is_gcf_tex_set) {
        cudaFreeArray(m_gcf_tex_arr);
        cudaDestroyTextureObject(m_gcf_tex);
    }

    cudaMallocArray(&m_gcf_tex_arr, &m_gcf_chan_desc, p_gcf_tex_dim, p_gcf_tex_dim);

    memset(&m_gcf_res_desc, 0, sizeof(m_gcf_res_desc));
    m_gcf_res_desc.resType = cudaResourceTypeArray;
    m_gcf_res_desc.res.array.array = m_gcf_tex_arr;
    // Specify texture object parameters
    // struct cudaTextureDesc tex_desc;
    memset(&m_gcf_tex_desc, 0, sizeof(m_gcf_tex_desc));
    m_gcf_tex_desc.addressMode[0] = cudaAddressModeClamp;
    m_gcf_tex_desc.addressMode[1] = cudaAddressModeClamp;
    // m_gcf_tex_desc.filterMode = cudaFilterModePoint; //cudaFilterModeLinear;
    m_gcf_tex_desc.filterMode = cudaFilterModeLinear;
    m_gcf_tex_desc.readMode = cudaReadModeElementType;
    m_gcf_tex_desc.normalizedCoords = 1;

    // m_gcf_res_desc.res.array.array = m_gcf_tex_arr;
    std::cout << "copying gcf\n";
    const size_t spitch = p_gcf_tex_dim * sizeof(float);
    cudaMemcpy2DToArray(m_gcf_tex_arr, 0, 0, p_gcf_2D_ptr, spitch, p_gcf_tex_dim * sizeof(float), p_gcf_tex_dim, cudaMemcpyHostToDevice);

    std::cout << "texture set\n";
    cudaCreateTextureObject(&m_gcf_tex, &m_gcf_res_desc, &m_gcf_tex_desc, NULL);

    is_gcf_tex_set = true;
    cuda_check_err(cudaPeekAtLastError());
}

void
MOFFCuHandler::create_gulp_custreams()
{
    cudaSetDevice(m_device_id);
    m_gulp_custreams.reset();
    m_gulp_custreams = std::make_unique<cudaStream_t[]>(m_nstreams);
    for (int i = 0; i < m_nstreams; ++i) {
        cudaStreamCreate(m_gulp_custreams.get() + i);
    }
}

void
MOFFCuHandler::reset_data(int p_nchan, size_t p_nseq_per_gulp, float* p_antpos_ptr, float* p_phases_ptr)
{
    cudaSetDevice(m_device_id);
    m_nseq_per_gulp = p_nseq_per_gulp;
    m_nchan_in = p_nchan;
    std::cout << "GPU resetting antpos\n";
    reset_antpos(p_nchan, p_antpos_ptr);
    std::cout << "GPU resetting phases\n";
    cuda_check_err(cudaPeekAtLastError());

    reset_phases(p_nchan, p_phases_ptr);
    cuda_check_err(cudaPeekAtLastError());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // test_gcf_texture<<<1, 1>>>(m_gcf_tex);
    cuda_check_err(cudaPeekAtLastError());
    cudaEventRecord(stop);
    cuda_check_err(cudaPeekAtLastError());
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Single grid time: " << milliseconds << std::endl;
    cudaDeviceSynchronize();
    cuda_check_err(cudaPeekAtLastError());
}

void
MOFFCuHandler::set_imaging_kernel()
{
    cudaSetDevice(m_device_id);
    assert(m_out_img_desc.img_size == HALF);
    if (m_out_img_desc.img_size == HALF) {
        m_imaging_kernel = (void*)block_fft_kernel<FFT64x64>;
        cudaFuncSetAttribute(
          m_imaging_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          FFT64x64::shared_memory_size);
        m_img_block_dim = FFT64x64::block_dim;
        m_shared_mem_size = FFT64x64::shared_memory_size;
    } else {
        // not implemented yet
    }
}

void
MOFFCuHandler::allocate_f_eng_gpu(size_t nbytes)
{
    cudaSetDevice(m_device_id);
    if (is_f_eng_cu_allocated) {
        cudaFree(m_f_eng_cu);
        is_f_eng_cu_allocated = false;
    }
    cudaMalloc(&m_f_eng_cu, nbytes);
    m_f_eng_bytes = nbytes;
    is_f_eng_cu_allocated = true;
}

void
MOFFCuHandler::allocate_out_img(size_t p_nbytes)
{
    cudaSetDevice(m_device_id);
    if (is_out_mem_set) {
        cudaFree(m_output_cu);
        is_out_mem_set = false;
    }
    cudaMalloc(&m_output_cu, p_nbytes);
    is_out_mem_set = true;
    m_out_img_bytes = p_nbytes;
}

void
MOFFCuHandler::set_img_grid_dim()
{
    cudaSetDevice(m_device_id);
    assert((void("Number of channels per stream cannot be zero"), m_nchan_per_stream > 0));
    if (m_nchan_per_stream > 0) {
        m_img_grid_dim = dim3(m_nchan_per_stream, 1, 1);
    }
}

void
MOFFCuHandler::process_gulp(uint8_t* p_data_ptr, float* p_out_ptr, bool p_first, bool p_last)
{
    cudaSetDevice(m_device_id);
    for (int i = 0; i < m_nstreams; ++i) {
        int f_eng_dat_offset = i * m_nbytes_f_eng_per_stream;
        int output_img_offset = i * m_nbytes_out_img_per_stream;
        auto stream_i = *(m_gulp_custreams.get() + i);
        int chan_offset = i * m_nchan_per_stream;

        void* args[] = {
            &m_f_eng_cu, &m_antpos_cu, &m_phases_cu, &m_nseq_per_gulp, &m_nchan_in, &m_gcf_tex, &m_output_cu, &chan_offset, &p_first
        };

        cudaMemcpyAsync(m_f_eng_cu + f_eng_dat_offset, p_data_ptr + f_eng_dat_offset, m_nbytes_f_eng_per_stream, cudaMemcpyHostToDevice, stream_i);

        cudaLaunchKernel(m_imaging_kernel, m_img_grid_dim, m_img_block_dim, args, m_shared_mem_size, stream_i);

        if (p_last) {
            cudaMemcpyAsync(p_out_ptr + output_img_offset, m_output_cu + output_img_offset, m_nbytes_out_img_per_stream, cudaMemcpyDeviceToHost, stream_i);
        }
    }

    if (p_last) {
        for (int i = 0; i < m_nstreams; ++i) {
            cudaStreamSynchronize(*(m_gulp_custreams.get() + i));
        }
        cuda_check_err(cudaPeekAtLastError());
    }
}

void
MOFFCuHandler::destroy_textures(cudaArray_t& p_tex_arr, cudaTextureObject_t& p_tex_obj)
{
    cudaSetDevice(m_device_id);
    cudaFreeArray(p_tex_arr);
    cudaDestroyTextureObject(p_tex_obj);
}

MOFFCuHandler::~MOFFCuHandler()
{
    cudaSetDevice(m_device_id);
    // destroy_textures(m_antpos_tex_arr, m_antpos_tex);
    // destroy_textures(m_phases_tex_arr, m_phases_tex);
    if (is_antpos_set) {
        cudaFree(m_antpos_cu);
    }
    if (is_phases_set) {
        cudaFree(m_phases_cu);
    }
}
