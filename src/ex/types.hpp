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

#ifndef SRC_EX_TYPES_HPP_
#define SRC_EX_TYPES_HPP_

#include <endian.h>

#include <any>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <variant>

#include "./constants.h"

/**
 * @brief Complex nibble. Independent of the host's endianness.
 *
 */
struct __attribute__((aligned(1))) CNib {
#if __BYTE_ORDER == __BIG_ENDIAN
  signed char im : 4, re : 4;
#elif __BYTE_ORDER == __LITTLE_ENDIAN
  signed char im : 4, re : 4;
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
struct __attribute__((aligned(2))) CNib2 {
  CNib X, Y;
};

/// Python dict-like data structure to describe Meta data
typedef std::unordered_map<std::string,
                           std::variant<int, int64_t, uint64_t, uint8_t,
                                        uint16_t, double, float, std::string>>
    dict_t;

// typedef std::unordered_map<std::string, std::any> dict_t;

struct MOFFCorrelatorDesc {
  /// @brief Accumulation (integration) time in ms
  float accum_time_ms{40};
  int nseq_per_gulp{1000};
  IMAGING_POL_MODE pol_mode{DUAL_POL};
  IMAGE_SIZE img_size{FULL};
  float grid_res_deg{1};
  int support_size{3};
  bool is_remove_autocorr{false};
  /// @brief Number of streams to split a gulp into. Can be at most
  /// MAX_GULP_STREAMS
  int nstreams{1};
  int nchan_out{128};
  int gcf_kernel_dim{40};  // decimeters
  unsigned int device_id{0};
  int nbuffers{20};
  int buf_size{64 * 64 * 132 /*chan*/ * sizeof(float) * 2 /*complex*/ *
               4 /*pols*/};
  bool page_lock_bufs{true};
  int max_tries_acq_buf{5};
  int kernel_oversampling_factor{2};
  bool use_bf16_accum{false};
};

struct OutImgDesc {
  IMAGING_POL_MODE pol_mode{DUAL_POL};
  IMAGE_SIZE img_size{HALF};
  int nchan_out;
};

struct FFTDxDesc {
  uint8_t* f_eng_g;
  float* antpos_g;
  float* phases_g;
  int nseq_per_gulp{1000};
  int nchan;
  cudaTextureObject_t gcf_tex;
  float* output_g;
  int chan_offset{0};
  bool is_first_gulp = true;
};

#endif  // SRC_EX_TYPES_HPP_
