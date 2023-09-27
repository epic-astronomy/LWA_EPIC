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

#ifndef SRC_EX_HELPER_TRAITS_HPP_
#define SRC_EX_HELPER_TRAITS_HPP_

#include <glog/logging.h>
#include <hwy/highway.h>

#include <cmath>
#include <memory>
#include <raftmanip>
#include <type_traits>

#include "./constants.h"
#include "./formats.h"

namespace hn = hwy::HWY_NAMESPACE;
// using tag8 = hn::ScalableTag<uint8_t>;

template <class T>
struct is_unique_ptr : std::false_type {};

template <class T, class D>
struct is_unique_ptr<std::unique_ptr<T, D>> : std::true_type {};

template <size_t A>
struct assert_gt_zero {
  static_assert(A > 0, "LTE to zero");
  static constexpr bool value = (A > 0);
};

/**
 * @brief Expression template to determine the nearest aligned buffer size
 *
 * @tparam Dtype Data type
 * @param p_buf_size Original number of elements
 * @return size_t Number of elements in the aligned buffer
 */
template <typename Dtype>
size_t nearest_integral_vec_size(size_t p_buf_size) {
  size_t lanes = HWY_LANES(Dtype);
  DLOG(INFO) << "The nearest lane-integral size of the buffer" << p_buf_size
             << ": " << ceil(static_cast<double>(p_buf_size) / lanes) * lanes;
  return ceil(static_cast<double>(p_buf_size) / lanes) * lanes;
}

constexpr int int_ceil(float f) {
  const int i = static_cast<int>(f);
  return f > i ? i + 1 : i;
}

/**
 * @brief Expression template to determine the offset required to align the data
 * part of the received packet
 *
 * @tparam Hdr Type of the packet header
 * @tparam T Data type
 * @tparam ExtraOffset Additional offset to account for if any
 */
template <typename Hdr, typename T, int ExtraOffset = 0>
struct AlignmentOffset {
 private:
  static constexpr size_t _nlanes = HWY_LANES(T);
  static constexpr size_t _hdr_size = sizeof(Hdr);
  static constexpr bool _valid_lanes = assert_gt_zero<_nlanes>::value;
  static constexpr bool _valid_hdr = assert_gt_zero<_hdr_size>::value;

 public:
  static constexpr int value =
      int_ceil((_hdr_size + ExtraOffset) / static_cast<double>(_nlanes)) *
          _nlanes -
      (_hdr_size + ExtraOffset);
};

template struct AlignmentOffset<chips_hdr_type, uint8_t>;
template struct AlignmentOffset<chips_hdr_type, uint8_t, 42>;
using VerbsOffset_t = AlignmentOffset<chips_hdr_type, uint8_t, 42>;
using ChipsOffset_t = AlignmentOffset<chips_hdr_type, uint8_t>;

template <size_t N>
using RftAffinityGrp = raft::parallel::affinity_group<N>;

template <size_t N>
using RftDeviceCpu = raft::parallel::device<raft::parallel::cpu, N>;

template <size_t CPUID, size_t AffGrpID>
using RftManip = raft::manip<RftAffinityGrp<AffGrpID>, RftDeviceCpu<CPUID>>;

/// @brief Test if the Buffer class can be constructed using a config object
/// @tparam Buf
/// @tparam
template <typename Buf, typename = void>
struct HasConfig : std::false_type {};

template <typename Buf>
struct HasConfig<Buf, std::void_t<typename Buf::config_t>> : std::true_type {};

#endif  // SRC_EX_HELPER_TRAITS_HPP_
