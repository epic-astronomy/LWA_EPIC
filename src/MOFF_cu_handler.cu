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
    m_gulp_custreams.reset();
    m_gulp_custreams = std::make_unique<cudaStream_t[]>(m_nstreams);
    for (int i = 0; i < m_nstreams; ++i) {
        cudaStreamCreate(m_gulp_custreams.get() + i);
    }
}

void
MOFFCuHandler::reset_data(int p_nchan, size_t p_nseq_per_gulp, float* p_antpos_ptr, float* p_phases_ptr)
{
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
    assert((void("Number of channels per stream cannot be zero"), m_nchan_per_stream > 0));
    if (m_nchan_per_stream > 0) {
        m_img_grid_dim = dim3(m_nchan_per_stream, 1, 1);
    }
}

void
MOFFCuHandler::process_gulp(uint8_t* p_data_ptr, float* p_out_ptr, bool p_first, bool p_last)
{
    for (int i = 0; i < m_nstreams; ++i) {
        int f_eng_dat_offset = i * m_nbytes_f_eng_per_stream;
        int output_img_offset = i * m_nbytes_out_img_per_stream;
        auto stream_i = *(m_gulp_custreams.get() + i);
        int chan_offset = i * m_nchan_per_stream;

        void* args[] = {
            &m_f_eng_cu, &m_antpos_cu, &m_phases_cu, &m_nseq_per_gulp, &m_nchan_in,
            &m_gcf_tex,
            &m_output_cu,
            &chan_offset,
            &p_first
        };

        cudaMemcpyAsync(m_f_eng_cu + f_eng_dat_offset, p_data_ptr + f_eng_dat_offset, m_nbytes_f_eng_per_stream, cudaMemcpyHostToDevice, stream_i);

        cudaLaunchKernel(m_imaging_kernel, m_img_grid_dim, m_img_block_dim, args, m_shared_mem_size, stream_i);

        if(p_last){
            cudaMemcpyAsync(p_out_ptr + output_img_offset, m_output_cu + output_img_offset, m_nbytes_out_img_per_stream, cudaMemcpyDeviceToHost, stream_i);
        }
    }

    cuda_check_err(cudaPeekAtLastError());
    if(p_last){
        cudaDeviceSynchronize();
    }
}


void
MOFFCuHandler::destroy_textures(cudaArray_t& p_tex_arr, cudaTextureObject_t& p_tex_obj)
{
    cudaFreeArray(p_tex_arr);
    cudaDestroyTextureObject(p_tex_obj);
}

MOFFCuHandler::~MOFFCuHandler()
{
    // destroy_textures(m_antpos_tex_arr, m_antpos_tex);
    // destroy_textures(m_phases_tex_arr, m_phases_tex);
    if (is_antpos_set) {
        cudaFree(m_antpos_cu);
    }
    if (is_phases_set) {
        cudaFree(m_phases_cu);
    }
}



// void
// MOFFCuHandler::process_gulp_old(uint8_t* p_data_ptr, size_t p_buf_size, float* p_out_ptr, size_t p_out_size, bool p_first = true, bool p_last = false)
// {
//     // auto data_ptr = p_payload.get_mbuf().get_data_ptr();
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);
//     cudaMemcpy(m_f_eng_cu, p_data_ptr, p_buf_size, cudaMemcpyHostToDevice);
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     std::cout << "Memcpy time: " << milliseconds << std::endl;
//     std::cout << "launch feng: " << int(p_data_ptr[0]) << " " << int(p_data_ptr[1]) << "\n";
//     int numblocks;
//     cuda_check_err(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
//       &numblocks, block_fft_kernel<FFT64x64>, int(FFT64x64::block_dim.x * FFT64x64::block_dim.y), size_t(FFT64x64::shared_memory_size)));
//     std::cout << "Max active blocks per SM: " << numblocks << std::endl;
//     std::cout << "Suggested threads per block: " << FFT64x64::suggested_ffts_per_block << std::endl;
//     // auto img_data = ImagingData(m_f_eng_cu, m_antpos_cu, m_phases_cu, 1000, 132, LWA_SV_NSTANDS, 2);
//     // // ImagingData<>* cu_ptr;
//     // cudaMalloc(&cu_ptr, sizeof(ImagingData<>));
//     // cudaMemcpy(cu_ptr, &img_data,sizeof(ImagingData<>), cudaMemcpyHostToDevice );

//     cudaFuncSetAttribute(
//       block_fft_kernel<FFT64x64>,
//       cudaFuncAttributeMaxDynamicSharedMemorySize,
//       FFT64x64::shared_memory_size);

//     void* args[] = {
//         &m_f_eng_cu, &m_antpos_cu, &m_phases_cu, &m_nseq_per_gulp, &m_nchan_in,
//         // &img_data,
//         &m_gcf_tex
//     };

//     using complex_type = typename FFT64x64::value_type;
//     // complex_type* workspace;
//     // cudaMalloc((void**)&workspace, 132 * FFT64x64::block_dim.x * FFT64x64::block_dim.y * sizeof(complex_type));
//     cuda_check_err(cudaPeekAtLastError());
//     dim3 grid_dim{ 112, 1, 1 };
//     std::cout << "launching the kernel\n";
//     cuda_check_err(cudaPeekAtLastError());
//     // cudaEvent_t start, stop;
//     // cudaEventCreate(&start);
//     // cudaEventCreate(&stop);
//     cudaEventRecord(start);
//     block_fft_kernel<FFT64x64><<<grid_dim, FFT64x64::block_dim, FFT64x64::shared_memory_size>>>(
//       m_f_eng_cu,
//       m_antpos_cu,
//       m_phases_cu,
//       m_nseq_per_gulp,
//       m_nchan_in,
//       m_gcf_tex,
//       m_output_cu);
//     // cudaLaunchCooperativeKernel((void*)block_fft_kernel<FFT64x64>, grid_dim, FFT64x64::block_dim, args, FFT64x64::shared_memory_size);
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     cudaMemcpy(p_out_ptr, m_output_cu, p_out_size, cudaMemcpyDeviceToHost);
//     // float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     std::cout << "Single FFT time: " << milliseconds << std::endl;
//     cuda_check_err(cudaPeekAtLastError());
//     cudaDeviceSynchronize();
//     std::cout << "kernel ran\n";
//     // cudaFree(cu_ptr);
// }

// void
// MOFFCuHandler::reset_1D_texture(bool p_reallocate, cudaArray_t& p_tex_arr, cudaTextureObject_t& p_tex_obj, cudaChannelFormatDesc& p_chan_desc, cudaResourceDesc& p_res_desc, cudaTextureDesc& p_tex_desc, int p_width, int p_depth, float* p_host_ptr)
// {

//     if (p_reallocate) {
//         cudaFreeArray(p_tex_arr);
//         cudaDestroyTextureObject(p_tex_obj);
//     }

//     auto extent_alloc = make_cudaExtent(p_width, 0, p_depth);
//     auto extent_cpy = make_cudaExtent(p_width, 1, p_depth);

//     cudaMalloc3DArray(&p_tex_arr, &p_chan_desc, extent_alloc, cudaArrayLayered);
//     cudaMemcpy3DParms parms_3d = { 0 };

//     int nelements_per_texel = 0;
//     nelements_per_texel += p_chan_desc.x > 0 ? 1 : 0;
//     nelements_per_texel += p_chan_desc.y > 0 ? 1 : 0;
//     nelements_per_texel += p_chan_desc.z > 0 ? 1 : 0;
//     nelements_per_texel += p_chan_desc.w > 0 ? 1 : 0;

//     parms_3d.srcPtr = make_cudaPitchedPtr(p_host_ptr, p_width * nelements_per_texel * sizeof(float), p_width, 1);
//     parms_3d.kind = cudaMemcpyHostToDevice;
//     parms_3d.extent = extent_cpy;
//     parms_3d.dstArray = p_tex_arr;

//     cudaMemcpy3D(&parms_3d);

//     cudaCreateTextureObject(&p_tex_obj, &p_res_desc, &p_tex_desc, NULL);
// }

// void
// MOFFCuHandler::set_default_tex_res_desc(cudaResourceDesc& p_res_desc, cudaTextureDesc& p_tex_desc, cudaArray_t& p_tex_arr)
// {
//     memset(&p_res_desc, 0, sizeof(p_res_desc));
//     p_res_desc.resType = cudaResourceTypeArray;
//     p_res_desc.res.array.array = p_tex_arr;
//     // Specify texture object parameters
//     // struct cudaTextureDesc tex_desc;
//     memset(&p_tex_desc, 0, sizeof(p_tex_desc));
//     p_tex_desc.addressMode[0] = cudaAddressModeClamp;
//     p_tex_desc.addressMode[1] = cudaAddressModeClamp;
//     p_tex_desc.addressMode[2] = cudaAddressModeClamp;
//     p_tex_desc.filterMode = cudaFilterModePoint;
//     p_tex_desc.readMode = cudaReadModeElementType;
//     p_tex_desc.normalizedCoords = 0;
// }

// void
// MOFFCuHandler::reset_antpos_tex(int p_nchan, float* p_antpos_ptr)
// {
//     if (!is_antpos_tex_set) {
//         set_default_tex_res_desc(m_antpos_res_desc, m_antpos_tex_desc, m_antpos_tex_arr);
//     }

//     reset_1D_texture(is_antpos_tex_set, m_antpos_tex_arr, m_antpos_tex, m_antpos_chan_desc, m_antpos_res_desc, m_antpos_tex_desc, LWA_SV_NSTANDS, p_nchan, p_antpos_ptr);

//     is_antpos_tex_set = true;
// }

// void
// MOFFCuHandler::reset_phases_tex(int p_nchan, float* p_phases_ptr)
// {
//     if (!is_phases_tex_set) {
//         set_default_tex_res_desc(m_phases_res_desc, m_phases_tex_desc, m_phases_tex_arr);
//     }
//     reset_1D_texture(is_antpos_tex_set, m_phases_tex_arr, m_phases_tex, m_phase_chan_desc, m_phases_res_desc, m_phases_tex_desc, LWA_SV_NSTANDS, p_nchan*2, p_phases_ptr);

//     is_phases_tex_set = true;
// }



// struct useless
// {
//     __half2 x, y;
// };

// __global__ void
// test_gcf_texture(cudaTextureObject_t p_gcf)
// {
//     // printf("%f\n", tex2D<float>(p_gcf, 0.5, 0.5));
//     // printf("%f\n", tex2D<float>(p_gcf, 0.5, 0));
//     // printf("%f\n", tex2D<float>(p_gcf, 0.0, 0.5));
//     // printf("%f\n", tex2D<float>(p_gcf, 0, 0));
//     // printf("%f\n", tex2D<float>(p_gcf, 0, 1.2));
//     // printf("%f\n", tex2D<float>(p_gcf, 1.2, 1.2));
//     // printf("%f\n", tex2D<float>(p_gcf, 0, abs(-1.1)));

//     useless pixel;
//     pixel.x.x = 0;
//     pixel.x.y = 0;
//     pixel.y.x = 0;
//     pixel.y.y = 0;
//     cnib x = { (signed char)0, (signed char)-3 };
//     cnib y = { (signed char)3, (signed char)0 };
//     cnib2 xy = { x, y };
//     uint8_t* f_eng = reinterpret_cast<uint8_t*>(&xy);

//     float antpos[3] = { 2, 1, 0.5 };
//     float phases[4] = { 1, -1, 0, 1 };
//     int half_support = 4;

//     int nstands = 1;
//     int u = 2;
//     int v = 1;

//     // get_grid_value<8, useless, DUAL_POL>(pixel, u, v, f_eng, &antpos[0], &phases[0], p_gcf, nstands);

//     // printf("%f %f %f %f\n", __half2float(pixel.x.x), __half2float(pixel.x.y), __half2float(pixel.y.x), __half2float(pixel.y.y));
// }