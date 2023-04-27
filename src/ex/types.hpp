#ifndef TYPES_H
#define TYPES_H

#include "constants.h"
#include <any>
#include <cstdint>
#include <cstring>
#include <endian.h>
#include <unordered_map>
#include <string>

/**
 * @brief Complex nibble. Independent of the host's endianness.
 *
 */
struct __attribute__((aligned(1))) cnib
{
#if __BYTE_ORDER == __BIG_ENDIAN
    signed char im : 4, re : 4;
#elif __BYTE_ORDER == __LITTLE_ENDIAN
    signed char re : 4, im : 4;
#else
    static_assert(false, "Unkonwn endianness. Alien!");
#endif
};

/**
 * @brief Complex nibble vector with two members, X and Y,
 * one for each polarization.
 *
 * @relatesalso MOFFCuHandler
 */
struct __attribute__((aligned(2))) cnib2
{
    cnib X, Y;
};

/// Python dict-like data structure to describe Meta data
typedef std::unordered_map<std::string, std::any> dict_t;

struct MOFFCorrelatorDesc
{
    /// @brief Accumulation (integration) time in ms
    float accum_time_ms{ 40 };
    int nseq_per_gulp{ 1000 };
    IMAGING_POL_MODE pol_mode{ DUAL_POL };
    IMAGE_SIZE img_size{ FULL };
    float grid_res_deg{1};    
    int support_size{ 2 };
    bool is_remove_autocorr{ false };
    /// @brief Number of streams to split a gulp into. Can be at most MAX_GULP_STREAMS
    int nstreams{ 1 };
    int nchan_out{ 128 };
    int gcf_kernel_dim{8 };
    int device_id{ 0 };
    int nbuffers{ 20 };
    int buf_size{ 64 * 64 * 132 /*chan*/ * 4 /*floating precision*/ * 2 /*complex*/ * 4 /*pols*/ };
    bool page_lock_bufs{ true };
    int max_tries_acq_buf{ 5 };
};

struct OutImgDesc
{
    IMAGING_POL_MODE pol_mode{ DUAL_POL };
    IMAGE_SIZE img_size{ HALF };
    int nchan_out;
};

struct FFTDxDesc{
    uint8_t* f_eng_g;
    float* antpos_g;
    float* phases_g;
    int nseq_per_gulp{1000};
    int nchan;
    cudaTextureObject_t gcf_tex;
    float* output_g;
    int chan_offset{0};
    bool is_first_gulp=true;
};


#endif // TYPES_H