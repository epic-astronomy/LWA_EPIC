#ifndef CU_HELPERS_CUH
#define CU_HELPERS_CUH

#include "constants.h"
#include "types.hpp"
#include <cuda_fp16.h>
#include <iostream>
#include <stdexcept>
#include "host_helpers.h"


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
}

#endif // CU_HELPERS_CUH