#ifndef CU_HELPERS_CUH
#define CU_HELPERS_CUH

#include "constants.h"
#include "host_helpers.h"
#include "types.hpp"
#include <cuda_fp16.h>
#include <cufftdx.hpp>
#include <iostream>
#include <stdexcept>

/**
 * @brief Half-precision complex multipy scale
 *
 * Multiply two complex numbers and optinally scale it with a scalar. The multiplication
 * is an fma operation and uses the optimized half2 intrinsics.
 *
 * @param a First input complex value
 * @param b Second input complex value
 * @param scale
 * @return s* (A * B)
 */
__device__ inline __half2
__half2cms(__half2 a, __half2 b, __half scale = __half(1))
{
    return __hmul2(__halves2half2(scale, scale), __halves2half2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x));
}

/**
 * @brief Mixed-precision complex multiply and scale using built-in intrinsics
 *
 * Performs s*(A * B), where s is a scalar, A and B are two complex numbers
 *
 * @tparam T Type of output complex data
 * @param out[out] Output variable
 * @param a[in] Complex number of type float2
 * @param b[in] Complex number of type cnib
 * @param scale[in] Scale valie
 * @return void
 *
 * @relatesalso MOFFCuHandler
 */
template<typename T>
__device__ inline void
__cms_f(T& out, const float2 a, const cnib b, float& scale)
{
    out.x = __fmul_rz(scale, __fadd_rz(__fmul_rz(a.x, b.re), -__fmul_rz(a.y, b.im)));
    out.y = __fmul_rz(scale, __fadd_rz(__fmul_rz(a.x, b.im), __fmul_rz(a.y, b.re)));
}

/**
 * @brief Mixed-precision complex multiply using built-in intrinsics
 *
 *
 * @tparam T Type of the output complex data
 * @param out[out] Output variable
 * @param a[in] Complex value of type float2
 * @param b[in] Complex value of type cnib
 * @return void
 *
 * @relatesalso MOFFCuHandler
 */
template<typename T>
__device__ inline void
__cm_f(T& out, const float2& a, const cnib& b)
{
    out.x += __fadd_rz(__fmul_rz(a.x, b.re), -__fmul_rz(a.y, b.im));
    out.y += __fadd_rz(__fmul_rz(a.x, b.im), __fmul_rz(a.y, b.re));
}

/**
 * @brief Get a const pointer to a specific sequence from F-engine data
 *
 * @tparam Order Order of the data
 * @param f_eng Pointer to F-Engine data
 * @param gulp_idx 0-based time-index of the gulp
 * @param chan_idx Channel number of the sequence
 * @param ngulps Total number sequences in the gulp
 * @param nchan Total number of channels in each sequence
 * @return Pointer to the start of the sequence
 *
 * @relatesalso MOFFCuHandler
 */
template<PKT_DATA_ORDER Order>
__device__ const uint8_t*
get_f_eng_sample(const uint8_t* f_eng, size_t gulp_idx, size_t chan_idx, size_t nseqs, size_t nchan)
{
    if (Order == CHAN_MAJOR) {
        return f_eng + LWA_SV_INP_PER_CHAN * (chan_idx * nseqs + gulp_idx);
    }
    if (Order == TIME_MAJOR) {
        return f_eng + LWA_SV_INP_PER_CHAN * (gulp_idx * nchan + chan_idx);
    }
}

/**
 * @brief Get antenna positions for a given channel
 *
 * The antenna position array must have dimensions nchan, nant, 3
 *
 * @param ant_pos Pointer to the antenna position array
 * @param chan_idx 0-based channel index
 * @return Pointer to the antenna position sub-array
 *
 * @relatesalso MOFFCuHandler
 */
__device__ const float*
get_ant_pos(const float* ant_pos, size_t chan_idx)
{
    return ant_pos + chan_idx * LWA_SV_NSTANDS * 3 /*dimensions*/;
}

/**
 * @brief Get pointer to antenna phases for the specified channel
 *
 * Phases array must have dimensions nchan, nant, 2
 *
 * @param phases Pointer to the phases array
 * @param chan_idx 0-based channel index
 * @return const pointer to the phases array for `chan_idx`
 *
 * @relatesalso MOFFCuHandler
 */
__device__ const float*
get_phases(const float* phases, size_t chan_idx)
{
    return phases + chan_idx * LWA_SV_NSTANDS * 2 /*real imag*/;
};

/**
 * @brief Transpose a matrix by swapping its upper and lower triangles
 *
 *
 * @tparam FFT FFT built using cuFFTDx
 * @param thread_reg Thread registers
 * @param shared_mem Shared memory. Must have at least half the memory size of the matrix
 * @param norm Normalizing factor
 * @return __device__
 *
 * @note Although this transpose takes half the memory of the original matrix, it also
 * requires twice the number of shared mem reads and writes compared to a full transpose
 * within the shared memory.
 */
template<class FFT, std::enable_if_t<std::is_same<__half2, typename FFT::output_type::value_type>::value, bool> = true>
__device__ void
transpose_tri(typename FFT::value_type thread_reg[FFT::elements_per_thread],
              typename FFT::value_type* shared_mem,
              typename cufftdx::precision_of<FFT>::type norm = 1.)
{
    constexpr int stride = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;

    // copy the upper triangle into the shared memory
    for (int i = 0; i < FFT::elements_per_thread; ++i) {
        auto col = threadIdx.x + i * stride;
        if (col <= threadIdx.y) {
            continue;
        }
        // The index for the upper triangle element in the shared memory is the original index
        // offset by the left triangle and diagonal elements.
        int upper_smem_idx = threadIdx.y * cufftdx::size_of<FFT>::value + col - (threadIdx.y + 1) * (threadIdx.y + 2) * 0.5;
        shared_mem[upper_smem_idx] = thread_reg[i];
    }

    __syncthreads();

    // swap the elements in lower traingle with those in the shared memory
    for (int i = 0; i < FFT::elements_per_thread; ++i) {
        auto col = threadIdx.x + i * stride;
        if (threadIdx.y <= col) {
            continue;
        }
        // The index for the lower triangle element is the same as the upper triangle but
        // with the row and columns exchanged.
        int lower_smem_idx = col * cufftdx::size_of<FFT>::value + threadIdx.y - (col + 1) * (col + 2) * 0.5;
        auto _temp = thread_reg[i];
        thread_reg[i] = shared_mem[lower_smem_idx];
        shared_mem[lower_smem_idx] = _temp;
    }

    __syncthreads();

    // copy the lower traiangle into the upper triangle
    for (int i = 0; i < FFT::elements_per_thread; ++i) {

        auto col = threadIdx.x + i * stride;
        if (col <= threadIdx.y) {
            continue;
        }
        int upper_smem_idx = threadIdx.y * cufftdx::size_of<FFT>::value + col - (threadIdx.y + 1) * (threadIdx.y + 2) * 0.5;
        thread_reg[i] = shared_mem[upper_smem_idx];
    }

    // normalize
    if (float(norm) != 1) {
        for (int i = 0; i < FFT::elements_per_thread; ++i) {
            thread_reg[i].x *= __half2half2(norm);
            thread_reg[i].y *= __half2half2(norm);
        }
    }
};

#endif // CU_HELPERS_CUH