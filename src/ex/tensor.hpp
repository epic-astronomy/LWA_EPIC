#ifndef TENSOR
#define TENSOR

#include "hwy/aligned_allocator.h"
#include "hwy/highway.h"
#include "orm_types.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <numeric>
#include <stddef.h>
#include <type_traits>
#include <utility>
#include <glog/logging.h>

namespace hn = hwy::HWY_NAMESPACE;

/**
 * @brief Basic tensor (N-Dimensional array)
 *
 * @tparam _Tp Data type
 * @tparam NDims Number of dimensions for the array
 */
template<typename _Tp, size_t NDims>
class Tensor
{
    static_assert(std::is_arithmetic_v<_Tp>, "Tensor data type must be numeric");

  private:
    typename hwy::AlignedFreeUniquePtr<_Tp[]> m_aligned_data_uptr;

  protected:
    _Tp* m_data_ptr{ nullptr };
    std::array<int, NDims> m_dims;
    std::array<int, NDims> m_strides;
    size_t m_size;

  public:
    Tensor(auto... p_dims);
    void assign_data(_Tp* p_data_ptr);
    void dissociate_data();
    _Tp& at(auto... p_idx) const;
    _Tp* const get_data_ptr() { return m_data_ptr; }
    std::array<int, NDims> shape() { return m_dims; }
    size_t size() const { return m_size; }
    void allocate();
    ~Tensor() { m_data_ptr = nullptr; }
};

template<typename _Tp, size_t NDims>
Tensor<_Tp, NDims>::Tensor(auto... p_dims)
{
    static_assert((std::is_integral_v<decltype(p_dims)> & ...));
    static_assert(sizeof...(p_dims) == NDims, "Mismatched dimensions and number of specified sizes");
    m_dims = { p_dims... };

    // compute strides in a row-major order
    int idx = 0;
    std::for_each(m_dims.begin(), m_dims.end(), [&](size_t) mutable {
        m_strides[idx] = std::accumulate(
          m_dims.begin() + 1 + idx, m_dims.end(), 1, std::multiplies<int>{});

        ++idx;
    });

    m_size = std::accumulate(m_dims.begin(), m_dims.end(), 1, std::multiplies<int>{});

    for (auto it : m_strides) {
        // std::cout << it << "\n";
    }
}

/**
 * @brief Assign a data pointer to the array
 *
 * @tparam _Tp Data type
 * @tparam NDims Number of array dimensions
 * @param p_data_ptr Pointer to the data
 */
template<typename _Tp, size_t NDims>
void
Tensor<_Tp, NDims>::assign_data(_Tp* p_data_ptr)
{
    m_data_ptr = p_data_ptr;
}

template<typename _Tp, size_t NDims>
void
Tensor<_Tp, NDims>::dissociate_data()
{
    m_data_ptr = nullptr;
}

/**
 * @brief
 *
 * @tparam _Tp Data type
 * @tparam NDims Number of array dimensions
 * @param p_idx Return a reference to the tensor element at the specified index. Must have the same size as the number of dimensions
 * @return _Tp&
 */
template<typename _Tp, size_t NDims>
_Tp&
Tensor<_Tp, NDims>::at(auto... p_idx) const
{
    static_assert((std::is_integral_v<decltype(p_idx)> & ...), "Indices must be integral types");
    static_assert(sizeof...(p_idx) == NDims, "Mismatched dimensions and number of specified sizes");
    assert(m_data_ptr != nullptr && "Cannot return data from an uninitialized tensor");

    size_t idx = 0;
    return m_data_ptr[((p_idx * m_strides[idx++]) + ...)];
}

/**
 * @brief Allocate memory for the tensor
 *
 * @tparam _Tp Data type
 * @tparam NDims Number of array dimensions
 */
template<typename _Tp, size_t NDims>
void
Tensor<_Tp, NDims>::allocate()
{

    m_aligned_data_uptr = std::move(hwy::AllocateAligned<_Tp>(m_size));
    m_data_ptr = m_aligned_data_uptr.get();
}

constexpr size_t EPICImgDim = 4 /*nchan, nx, ny, npol*/;

/**
 * @brief Pseudo Stokes tensor class tailored for use with EPIC
 *
 * Each pixel value (4-pols) is treated as a 128-bit vector for fast
 * computation. Provides an interface for n-channel summation, and
 * source pixel extraction
 *
 * @tparam _Tp Data type
 */
template<typename _Tp = float>
class PSTensor : public Tensor<_Tp, EPICImgDim>
{
    static constexpr int NSTOKES{ 4 };
    using tag_t = hn::FixedTag<_Tp, NSTOKES>;
    using vec_t = hn::Vec<tag_t>;
    tag_t _stokes_tag;

    size_t m_xdim;
    size_t m_ydim;
    size_t m_npix_per_img;
    size_t m_nchan;

    size_t chan_pix2idx(size_t p_chan_id, size_t p_pix)
    {
        return (p_chan_id * m_npix_per_img + p_pix) * NSTOKES;
    }

  public:
    /**
     * @brief Construct a new PSTensor object
     *
     * @param p_nchan Number of channels in the image
     * @param p_xdim Image X-dimensions
     * @param p_ydim Image Y-dimensions
     */
    PSTensor(int p_nchan, int p_xdim, int p_ydim)
      : Tensor<_Tp, EPICImgDim>(p_nchan, p_xdim, p_ydim, NSTOKES)
      , m_nchan(p_nchan)
      , m_xdim(p_xdim)
      , m_ydim(p_ydim)
      , m_npix_per_img(p_xdim * p_ydim)
    {
    }

    /**
     * @brief Return a vector (all pols) at the specified flat index
     *
     * @param p_idx
     * @return vec_t
     */
    inline vec_t operator[](size_t p_idx) const
    {
        return hn::Load(
          _stokes_tag, this->m_data_ptr + p_idx);
    }

    /**
     * @brief In-place addition operator
     *
     * @param rhs Tensor to add in-place
     * @return PSTensor
     */
    PSTensor& operator+=(const PSTensor& rhs)
    {
        assert((this->size() == rhs.size()) && "Input sizes do not match. Cannot add the tensors");

        size_t nelems = rhs.size();
        for (size_t i = 0; i < nelems; i += NSTOKES) {
            hn::Store((*this)[i] + rhs[i],
                      _stokes_tag,
                      this->m_data_ptr + i);
        }

        return *this;
    }

    /**
     * @brief Return a vector (all pols) for the specified channel and pixel
     *
     * @param p_chan_id Channel number
     * @param p_xpix 0-based x-position of the pixel
     * @param p_ypix 0-based y-position of the pixel
     * @return vec_t
     */
    vec_t operator()(size_t p_chan_id, size_t p_xpix, size_t p_ypix) const
    {
        assert(this->m_data_ptr != nullptr && "Cannot return data from an uninitialized tensor");

        return hn::Load(
          _stokes_tag, &(this->at(p_chan_id, p_xpix, p_ypix, 0)));
    }

    /**
     * @brief Set the pixel value (all pols) in the specified channel
     *
     * @param p_chan_id 0-based channel number
     * @param p_xpix 0-based x-position the pixel
     * @param p_ypix 0-based y-position of the pixel
     * @param p_vec All pol vector to be set
     */
    void set(size_t p_chan_id, size_t p_xpix, size_t p_ypix, const vec_t& p_vec)
    {
        hn::Store(
          p_vec, _stokes_tag, &(this->at(p_chan_id, p_xpix, p_ypix, 0)));
    }

    /**
     * @brief Set a pixel based at the specified flat index in an image wthin a channel
     *
     * @param p_chan_id 0-based channel number
     * @param p_pix Flat index of the pixel with in a single image
     * @param p_vec Vector to be set at the pixel
     */
    void set(size_t p_chan_id, size_t p_pix, const vec_t& p_vec)
    {
        hn::Store(
          p_vec, _stokes_tag, this->m_data_ptr + chan_pix2idx(p_chan_id, p_pix));
    }

    /**
     * @brief Set a pixel at the specified flat index
     *
     * @param p_idx Flat index position of the pixel
     * @param p_vec Vector to the stored
     */
    void set(size_t p_idx, const vec_t& p_vec)
    {
        hn::Store(p_vec, _stokes_tag, this->m_data_ptr + p_idx);
    }

    /**
     * @brief Sum channels based on the specified output tensor.
     *
     * The function calculates how many input channels must be summed to creat one outpuc channel
     *
     * @param p_out_tensor Output tensor where the channel summation will be stored
     */
    void combine_channels(PSTensor<_Tp>& p_out_tensor)
    {
        auto out_nchan = p_out_tensor.shape()[0];
        assert((this->m_dims[0] % out_nchan == 0) && "Input channels must be an integral multiple of output channels");
        int ncombine = this->m_dims[0] / out_nchan;

        auto out_data_ptr = p_out_tensor.get_data_ptr();

        // very ugly.
        // for (size_t i = 0; i < ncombine; ++i) {
        //     for (size_t chan = 0; chan < out_nchan; ++chan) {
        //         for (size_t pix = 0; pix < m_npix_per_img; ++pix) {
        //             auto idx_in = (chan * ncombine + i) * m_npix_per_img + pix;
        //             idx_in *= NSTOKES;
        //             auto idx_out = chan * m_npix_per_img + pix;
        //             idx_out *= NSTOKES;
        //             printf("%lu %lu %lu %lu %lu \n", i, chan, pix, idx_in, idx_out);
        //             if (i == 0) {
        //                 hn::Store(
        //                   hn::Load(_stokes_tag, this->m_data_ptr + idx_in), _stokes_tag, out_data_ptr + idx_out);
        //             } else {
        //                 hn::Store(
        //                   hn::Load(_stokes_tag, out_data_ptr + idx_out) + hn::Load(_stokes_tag, this->m_data_ptr + idx_in), _stokes_tag, out_data_ptr + idx_out);
        //             }
        //         }
        //     }
        // }

        for (size_t chan = 0; chan < out_nchan; ++chan) {
            // for (size_t slice = 0; slice < ncombine; ++slice) {
            //     add_chan_slice(
            //       p_out_tensor, chan * ncombine + slice, chan, (slice == 0 ? true : false));
            // }
            add_chan_nslices(p_out_tensor, chan, ncombine);
        }
    }

    /**
     * @brief Add the specified number of input channels for each output channels and store it in the output tensor
     *
     * @param p_out_tensor Output Tensor
     * @param p_chan_out Number of output channels
     * @param p_nslices Number of input channels per output channel
     */
    void add_chan_nslices(PSTensor<_Tp>& p_out_tensor, size_t p_chan_out, size_t p_nslices)
    {
        for (size_t slice = 0; slice < p_nslices; ++slice) {
            add_chan_slice(
              p_out_tensor, p_chan_out * p_nslices + slice, p_chan_out, (slice == 0 ? true : false));
        }
    }

    /**
     * @brief Add a specified input channel to the output channel
     *
     * @param p_out_tensor Out Tensor
     * @param chan_in 0-based input channel number
     * @param chan_out 0-based output channel number
     * @param p_assign Flag to indicate whether to assign or sum-in-place the channel
     */
    void add_chan_slice(PSTensor<_Tp>& p_out_tensor, size_t chan_in, size_t chan_out, bool p_assign = false)
    {
        auto out_ptr = p_out_tensor.get_data_ptr();
        for (size_t i = 0; i < m_npix_per_img; ++i) {
            auto idx_in = chan_pix2idx(chan_in, i);
            auto idx_out = chan_pix2idx(chan_out, i);

            auto out_vec = (p_assign ? Zero(_stokes_tag) : p_out_tensor[idx_out]) + (*this)[idx_in];

            p_out_tensor.set(idx_out, out_vec);
            // hn::Store(
            //   out_vec , _stokes_tag, out_ptr + idx_out);
        }
    }

    void extract_pixels(const EpicPixelTableMetaRows& p_meta, float* p_pixels)
    {
        for (int i = 0; i < p_meta.pixel_coords_sft.size(); ++i) {//coord loop
            for (int j = 0; j < m_nchan; ++j) {//chan loop
                int x = p_meta.pixel_coords_sft[i].first;
                int y = p_meta.pixel_coords_sft[i].second;
                hn::Store(
                  (*this)(j, x, y), _stokes_tag, p_pixels + (i * m_nchan + j) * NSTOKES);
            }//chan loop
        }//coord loop
    }
};

#endif /* TENSOR */
