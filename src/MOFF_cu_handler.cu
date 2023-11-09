/*
 Copyright (c) 2023 The EPIC++ authors

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 the Software, and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "ex/MOFF_cu_handler.h"
#include "ex/constants.h"
#include "ex/cu_helpers.cuh"
#include "ex/fft_dx.cuh"
#include "ex/gridder.cuh"
#include "ex/types.hpp"
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <iostream>
#include <nvml.h>
#include <glog/logging.h>

namespace cg = cooperative_groups;
typedef __half2 f16_accum_t;

// Note: For a 40 ms gulp, bfloat takes about a 36.8 ms while half takes about 38 ms.
// However, the loss of precision is larger with bfloat compared to half.
// Hence half is chosen for on-chip accumulation.
// typedef __nv_bfloat162 f16_accum_t;

void
MOFFCuHandler::ResetAntpos(int p_nchan, float* p_antpos_ptr)
{
    cuda_check_err(cudaSetDevice(m_device_id));
    if (is_antpos_set) {
        cudaFree(m_antpos_cu);
    }
    auto nbytes = LWA_SV_NSTANDS * p_nchan * 3 * sizeof(float);
    cudaMalloc(&m_antpos_cu, nbytes);
    cudaMemcpy(m_antpos_cu, p_antpos_ptr, nbytes, cudaMemcpyHostToDevice);
    is_antpos_set = true;
}

void
MOFFCuHandler::ResetPhases(int p_nchan, float* p_phases_ptr)
{
    cuda_check_err(cudaSetDevice(m_device_id));
    if (is_phases_set) {
        cudaFree(m_phases_cu);
    }

    auto nbytes = LWA_SV_NSTANDS * p_nchan * LWA_SV_NPOLS * 2 * sizeof(float);
    cudaMalloc(&m_phases_cu, nbytes);
    cudaMemcpy(m_phases_cu, p_phases_ptr, nbytes, cudaMemcpyHostToDevice);

    is_phases_set = true;
}

void
MOFFCuHandler::ResetGcfTex(int p_gcf_tex_dim, float* p_gcf_2D_ptr)
{
    cuda_check_err(cudaSetDevice(m_device_id));
    if (is_gcf_tex_set) {
        cuda_check_err(cudaFreeArray(m_gcf_tex_arr));
        cuda_check_err(cudaDestroyTextureObject(m_gcf_tex));
    }

    cudaMallocArray(&m_gcf_tex_arr, &m_gcf_chan_desc, p_gcf_tex_dim, p_gcf_tex_dim);

    memset(&m_gcf_res_desc, 0, sizeof(m_gcf_res_desc));
    m_gcf_res_desc.resType = cudaResourceTypeArray;
    m_gcf_res_desc.res.array.array = m_gcf_tex_arr;
    // Specify texture object parameters
    memset(&m_gcf_tex_desc, 0, sizeof(m_gcf_tex_desc));
    m_gcf_tex_desc.addressMode[0] = cudaAddressModeClamp;
    m_gcf_tex_desc.addressMode[1] = cudaAddressModeClamp;
    // m_gcf_tex_desc.filterMode = cudaFilterModePoint;
    m_gcf_tex_desc.filterMode = cudaFilterModeLinear;
    m_gcf_tex_desc.readMode = cudaReadModeElementType;
    m_gcf_tex_desc.normalizedCoords = 0;

    VLOG(2) << "copying gcf\n";
    const size_t spitch = p_gcf_tex_dim * sizeof(float);
    cudaMemcpy2DToArray(m_gcf_tex_arr, 0, 0, p_gcf_2D_ptr, spitch, p_gcf_tex_dim * sizeof(float), p_gcf_tex_dim, cudaMemcpyHostToDevice);

    VLOG(2) << "texture set\n";
    cudaCreateTextureObject(&m_gcf_tex, &m_gcf_res_desc, &m_gcf_tex_desc, NULL);

    is_gcf_tex_set = true;
    cuda_check_err(cudaPeekAtLastError());
}

void
MOFFCuHandler::CreateGulpCuStreams()
{
    cuda_check_err(cudaSetDevice(m_device_id));
    m_gulp_custreams.reset();
    m_gulp_custreams = std::make_unique<cudaStream_t[]>(m_nstreams);
    for (int i = 0; i < m_nstreams; ++i) {
        cudaStreamCreate(m_gulp_custreams.get() + i);
    }
}

void MOFFCuHandler::ResetGcfElem(int p_nchan, int p_support, int p_chan0, float p_delta, int p_grid_size){
    cuda_check_err(cudaSetDevice(m_device_id));
    if(is_m_gcf_elem_set){
        cuda_check_err(cudaFree(m_gcf_elem));
        is_m_gcf_elem_set = false;
    }
    auto nelements_gcf = (p_support) * (p_support);
    auto nbytes = LWA_SV_NSTANDS * p_nchan * nelements_gcf * sizeof(float);
    cuda_check_err(cudaMalloc(&m_gcf_elem, nbytes));
    is_m_gcf_elem_set=true;

    int block_size = (MAX_THREADS_PER_BLOCK/float(nelements_gcf)) ;
    block_size *= nelements_gcf;

    VLOG(2)<<"Pre-computing GCF elements\n"<<p_support<<" "<<block_size<<" "<<nelements_gcf;
    ComputeGcfElements<<<p_nchan, block_size>>>(m_gcf_elem, m_antpos_cu, p_chan0, p_delta, m_gcf_tex,p_grid_size, (p_support), LWA_SV_NSTANDS);
    VLOG(2)<<"Done\n";

    cudaDeviceSynchronize();
    cuda_check_err(cudaPeekAtLastError());
}

void MOFFCuHandler::GetCorrectionKernel(float* p_out_kernel, int p_support, int p_nchan){
    cuda_check_err(cudaSetDevice(m_device_id));
    if(m_nchan_in==0){
        LOG(FATAL)<<"Number of input channels is not set. Unable to compute the averaged kernel\n";
    }
    int nbytes = p_support * p_support * p_nchan * sizeof(float);
    if(is_correction_kernel_set){
        cudaFree(m_correction_kernel_d);
        is_correction_kernel_set = false;
    }
    cuda_check_err(cudaMalloc(&m_correction_kernel_d, nbytes));
    is_correction_kernel_set = true;

    

    ComputeAvgGriddingKernel<<<p_nchan, LWA_SV_NSTANDS>>>(m_gcf_elem, m_correction_kernel_d ,p_nchan, p_support);

    cudaMemcpy(p_out_kernel, m_correction_kernel_d, nbytes, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cuda_check_err(cudaPeekAtLastError());
}
//void MOFFCuHandler::SetCorrectionGrid(float* corr_grid);


void MOFFCuHandler::SetCorrectionGrid(float* p_in_correction_grid, int p_grid_size, int p_nchan){
    cuda_check_err(cudaSetDevice(m_device_id));
    int nbytes = p_grid_size * p_grid_size * p_nchan * sizeof(float);
    if(is_correction_grid_set){
        cudaFree(m_correction_grid_d);
        is_correction_grid_set=false;
    }
    cuda_check_err(cudaMalloc(&m_correction_grid_d, nbytes));
    is_correction_grid_set = true;

    cuda_check_err(cudaMemcpy(m_correction_grid_d, p_in_correction_grid, nbytes, cudaMemcpyHostToDevice));
    VLOG(2)<<"FINE3\n";
    cudaDeviceSynchronize();
    cuda_check_err(cudaPeekAtLastError());
}

void
MOFFCuHandler::ResetData(int p_nchan, size_t p_nseq_per_gulp, float* p_antpos_ptr, float* p_phases_ptr)
{
    cuda_check_err(cudaSetDevice(m_device_id));
    m_nseq_per_gulp = p_nseq_per_gulp;
    m_nchan_in = p_nchan;

    VLOG(2) << "GPU resetting antpos\n";
    ResetAntpos(p_nchan, p_antpos_ptr);
    cuda_check_err(cudaPeekAtLastError());

    VLOG(2) << "GPU resetting phases\n";
    ResetPhases(p_nchan, p_phases_ptr);
    cuda_check_err(cudaPeekAtLastError());

    cudaDeviceSynchronize();
    cuda_check_err(cudaPeekAtLastError());
}

void
MOFFCuHandler::SetImagingKernel()
{   int smemSize;
    cudaDeviceGetAttribute(&smemSize, cudaDevAttrMaxSharedMemoryPerBlock, m_device_id);
    VLOG(2)<<"Max shared memory per block: "<<smemSize<<" bytes\n";
    cuda_check_err(cudaSetDevice(m_device_id));
    // assert(m_out_img_desc.img_size == HALF);
    if (m_out_img_desc.img_size == HALF) {
        VLOG(3)<<"Setting the imaging kernel to 64x64\n";
        VLOG(3)<<"Shared memory size: "<<FFT64x64::shared_memory_size<<" bytes\n";
        VLOG(3)<<FFT64x64::block_dim.x<<" "<<FFT64x64::block_dim.y<<"\n";
        if(use_bf16_accum){
            m_imaging_kernel = GetImagingKernel<FFT64x64, f16_accum_t>(m_support_size);
        }else{
            m_imaging_kernel = GetImagingKernel<FFT64x64, float2>(m_support_size);
        }
        m_img_block_dim = FFT64x64::block_dim;
        m_shared_mem_size = FFT64x64::shared_memory_size*2;
    } else {
         VLOG(3)<<"Setting the imaging kernel to 128x128\n";
        VLOG(3)<<"Shared memory size: "<<FFT128x128::shared_memory_size<<" bytes "<<FFT128x128::elements_per_thread<<"\n";
        VLOG(3)<<FFT64x64::block_dim.x<<" "<<FFT128x128::block_dim.y<<"\n";


        if(use_bf16_accum){
            VLOG(4)<<"ACCUMULATING in 16BIT\n";
            m_imaging_kernel = GetImagingKernel<FFT128x128, f16_accum_t>(m_support_size);
        }else{
            m_imaging_kernel = GetImagingKernel<FFT128x128, float2>(m_support_size);
        }
        m_img_block_dim = FFT128x128::block_dim;
        m_shared_mem_size = FFT128x128::shared_memory_size*1.5;
    }
    cudaFuncSetAttribute(
          m_imaging_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          m_shared_mem_size);
}

void
MOFFCuHandler::AllocateFEngGpu(size_t nbytes)
{
    cuda_check_err(cudaSetDevice(m_device_id));
    if (is_f_eng_cu_allocated) {
        cudaFree(m_f_eng_cu);
        is_f_eng_cu_allocated = false;
    }
    cudaMalloc(&m_f_eng_cu, nbytes);
    m_f_eng_bytes = nbytes;
    is_f_eng_cu_allocated = true;
}

void
MOFFCuHandler::AllocateOutImg(size_t p_nbytes)
{
    cuda_check_err(cudaSetDevice(m_device_id));
    if (is_out_mem_set) {
        cudaFree(m_output_cu);
        is_out_mem_set = false;
    }
    cudaMalloc(&m_output_cu, p_nbytes);
    is_out_mem_set = true;
    m_out_img_bytes = p_nbytes;
}

void
MOFFCuHandler::SetImgGridDim()
{
    cuda_check_err(cudaSetDevice(m_device_id));
    assert((void("Number of channels per stream cannot be zero"), m_nchan_per_stream > 0));
    if (m_nchan_per_stream > 0) {
        m_img_grid_dim = dim3(m_nchan_per_stream, 1, 1);
    }
}

void
MOFFCuHandler::ProcessGulp(uint8_t* p_data_ptr, float* p_out_ptr, bool p_first, bool p_last, int p_chan0, float p_delta)
{
    cuda_check_err(cudaSetDevice(m_device_id));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    VLOG(2)<<"FEng bytes per stream: "<<m_nbytes_f_eng_per_stream<<". OutImg bytes per stream: "<<m_nbytes_out_img_per_stream<<". Chan per stream: "<<m_nchan_per_stream<<". NStreams: "<<m_nstreams<<"\n";
    VLOG(2)<<"Nseq per gulp: "<<m_nseq_per_gulp<<"\n";
    for (int i = 0; i < m_nstreams; ++i) {
        int f_eng_dat_offset = i * m_nbytes_f_eng_per_stream;
        int output_img_offset = i * m_nbytes_out_img_per_stream/sizeof(float);
        auto stream_i = *(m_gulp_custreams.get() + i);
        int chan_offset = i * m_nchan_per_stream;

        void* args[] = {
            &m_f_eng_cu, &m_antpos_cu, &m_phases_cu, &m_nseq_per_gulp, &m_nchan_in, &m_gcf_tex, &m_output_cu, &chan_offset, &p_first, &m_gcf_elem, &m_correction_grid_d
        };

        cuda_check_err(
          cudaMemcpyAsync(
            (void*)(m_f_eng_cu + f_eng_dat_offset),
            (void*)(p_data_ptr + f_eng_dat_offset),
            m_nbytes_f_eng_per_stream,
            cudaMemcpyHostToDevice,
            stream_i));
        VLOG(2)<<"Launching the kernel\n";
        if(m_imaging_kernel==nullptr){
            VLOG(2)<<"Null imaging kernel\n";
        }
        VLOG(2)<<m_img_grid_dim.x<<" "<<m_img_grid_dim.y<<" "<<m_img_block_dim.x<<" "<<m_img_block_dim.y<<" "<<m_shared_mem_size<<std::endl;
        cuda_check_err(cudaLaunchKernel(m_imaging_kernel, m_img_grid_dim, m_img_block_dim, args, m_shared_mem_size, stream_i));

        VLOG(2)<<"chan0: "<<p_chan0<<" delta: "<<p_delta<<"\n";

        VLOG(2)<<i<<" "<<output_img_offset<<" "<<"\n";
        if (p_last) {
            cuda_check_err(cudaMemcpyAsync((void*)(p_out_ptr + output_img_offset), (void*)(m_output_cu + output_img_offset), m_nbytes_out_img_per_stream, cudaMemcpyDeviceToHost, stream_i));
        }
    }

    if (p_last) {
        VLOG(2)<<"Syncing the kernels\n";
        for (int i = 0; i < m_nstreams; ++i) {
            cuda_check_err(cudaStreamSynchronize(*(m_gulp_custreams.get() + i)));
        }
        VLOG(2)<<"Syncing done\n";
        cuda_check_err(cudaPeekAtLastError());
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    PrometheusExporter::ObserveRunTimeValue(m_gulp_exec_gauge_id, milliseconds);
    VLOG(1) << "Gulp processing time (ms) on GPU("<<m_device_id<<"): " << milliseconds << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

void
MOFFCuHandler::DestroyTextures(cudaArray_t& p_tex_arr, cudaTextureObject_t& p_tex_obj)
{
    cuda_check_err(cudaSetDevice(m_device_id));
    cudaFreeArray(p_tex_arr);
    cudaDestroyTextureObject(p_tex_obj);
}

MOFFCuHandler::~MOFFCuHandler()
{
    cuda_check_err(cudaSetDevice(m_device_id));
    // DestroyTextures(m_antpos_tex_arr, m_antpos_tex);
    // DestroyTextures(m_phases_tex_arr, m_phases_tex);
    if (is_antpos_set) {
        cudaFree(m_antpos_cu);
    }
    if (is_phases_set) {
        cudaFree(m_phases_cu);
    }

    if(is_correction_kernel_set){
        cudaFree(m_correction_kernel_d);
    }

    if(is_correction_grid_set){
        cudaFree(m_correction_grid_d);
    }

    if(is_m_gcf_elem_set){
        cudaFree(m_gcf_elem);
    }

    if(is_out_mem_set){
        cudaFree(m_output_cu);
    }

    if(is_f_eng_cu_allocated){
        cudaFree(m_f_eng_cu);
    }
}
