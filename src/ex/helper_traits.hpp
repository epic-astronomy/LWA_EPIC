#ifndef HELPER_TRAITS
#define HELPER_TRAITS

#include "formats.h"
// #include "hwy/aligned_allocator.h"
// #include "hwy/base.h"
#include "bf_ibverbs.hpp"
#include "constants.h"
#include "hwy/highway.h"
#include <cmath>
#include <glog/logging.h>
#include <memory>
#include <type_traits>

namespace hn = hwy::HWY_NAMESPACE;
// using tag8 = hn::ScalableTag<uint8_t>;

template<class T>
struct is_unique_ptr : std::false_type
{
};

template<class T, class D>
struct is_unique_ptr<std::unique_ptr<T, D>> : std::true_type
{
};

// template<typename T>
// bool
// is_in_bounds(const T& value, const T& low, const T& high)
// {
//     return !(value < low) && (value < high); // low<=val<high
// };

template<size_t A>
struct assert_gt_zero
{
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
template<typename Dtype>
size_t
nearest_integral_vec_size(size_t p_buf_size)
{
    size_t lanes = HWY_LANES(Dtype);
    DLOG(INFO) << "The nearest lane-integral size of the buffer" << p_buf_size << ": " << ceil(double(p_buf_size) / lanes) * lanes;
    return ceil(double(p_buf_size) / lanes) * lanes;
};

/**
 * @brief Expression template to determine the offset required to align the data part
 * of the received packet
 *
 * @tparam Hdr Type of the packet header
 * @tparam T Data type
 * @tparam ExtraOffset Additional offset to account for if any
 */
template<typename Hdr, typename T, int ExtraOffset = 0>
struct alignment_offset
{
  private:
    static constexpr size_t _nlanes = HWY_LANES(T);
    static constexpr size_t _hdr_size = sizeof(Hdr);
    static constexpr bool _valid_lanes = assert_gt_zero<_nlanes>::value;
    static constexpr bool _valid_hdr = assert_gt_zero<_hdr_size>::value;

  public:
    static constexpr int value = std::ceil((_hdr_size + ExtraOffset) / double(_nlanes)) * _nlanes - (_hdr_size + ExtraOffset);
};

template struct alignment_offset<chips_hdr_type, uint8_t>;
template struct alignment_offset<chips_hdr_type, uint8_t, BF_VERBS_PAYLOAD_OFFSET>;

#endif // HELPER_TRAITS