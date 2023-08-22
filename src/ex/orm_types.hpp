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

#ifndef SRC_EX_ORM_TYPES_HPP_
#define SRC_EX_ORM_TYPES_HPP_

#include <hwy/aligned_allocator.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "./types.hpp"

struct EpicPixelTableMetaRows {
  constexpr static int NSTOKES{4};
  std::string m_uuid;
  std::vector<int> source_ids;
  size_t nsrcs;
  size_t kernel_size;
  std::vector<std::pair<int, int>> pixel_coords;
  std::vector<std::pair<int, int>> pixel_coords_sft;
  std::vector<std::pair<float, float>> pixel_lm;
  // std::vector<std::pair<float, float>> pixel_skypos;
  std::vector<std::pair<int, int>> pixel_offst;

  size_t m_ncoords;
  int meta_version{-1};

  EpicPixelTableMetaRows(int ncoords, int n_sources) {
    nsrcs = n_sources;
    m_ncoords = ncoords;

    if (nsrcs == 0 || m_ncoords == 0) {
      return;
    }
    source_ids.reserve(n_sources);
    pixel_coords.reserve(ncoords);
    pixel_coords_sft.reserve(ncoords);
    pixel_lm.reserve(ncoords);
    // pixel_skypos.reserve(ncoords);
    pixel_offst.reserve(ncoords);

    assert(IsKernelSizeValid(n_sources, ncoords) &&
           "Invalid kernel size. Each source must have ksize * ksize number of "
           "sources.");

    kernel_size = ncoords / n_sources;
  }

  EpicPixelTableMetaRows() {}

  void TransformPixCoords(int xdim, int ydim) {
    for (size_t i = 0; i < m_ncoords; ++i) {
      int x = pixel_coords[i].first;
      // The array y-index starts from the top while the image from the bottom
      int y = ydim - 1 - pixel_coords[i].second;

      // transpose
      std::swap(x, y);

      // circshift
      x = (x + xdim / 2) % xdim;
      y = (y + ydim / 2) % ydim;
      pixel_coords_sft.push_back(std::pair<int, int>(x, y));
    }
  }

  bool diff(const EpicPixelTableMetaRows& rhs) {
    return (m_ncoords != rhs.m_ncoords) || (pixel_coords != rhs.pixel_coords) ||
           (pixel_lm != rhs.pixel_lm) ||
           //    (pixel_skypos != rhs.pixel_skypos) ||
           (pixel_offst != rhs.pixel_offst);
  }

  bool IsKernelSizeValid(int nsrcs, int ncoords) {
    size_t nkernel_elems = ncoords / nsrcs;
    size_t ksize = sqrt(nkernel_elems);
    return nkernel_elems == ksize * ksize;
  }
};

template <typename _Dtype>
struct EpicPixelTableDataRows : EpicPixelTableMetaRows {
  struct _config {
    int ncoords;
    int nsrcs;
    int nchan;
    bool CheckOpts() {
      bool flag = true;
      flag &= ncoords < 0 ? false : true;
      flag &= nchan < 0 ? false : true;
      flag &= nsrcs < 0 ? false : true;

      return true;
    }
  };

  using config_t = _config;
  size_t m_nchan{32};
  dict_t m_img_metadata;

  hwy::AlignedFreeUniquePtr<_Dtype[]> pixel_values;
  explicit EpicPixelTableDataRows(config_t config)
      : EpicPixelTableMetaRows(config.ncoords, config.nsrcs) {
    pixel_values = std::move(
        hwy::AllocateAligned<_Dtype>(config.ncoords * NSTOKES * config.nchan));
    m_nchan = config.nchan;
  }

  void copy_meta(const EpicPixelTableMetaRows& meta) {
    if (meta_version == meta.meta_version) {
      return;
    }

    if (m_ncoords != meta.m_ncoords) {
      pixel_values.reset();
      pixel_values = std::move(
          hwy::AllocateAligned<_Dtype>(meta.m_ncoords * NSTOKES * m_nchan));
    }
    pixel_coords = meta.pixel_coords;
    pixel_lm = meta.pixel_lm;
    // pixel_skypos = meta.pixel_skypos;
    pixel_offst = meta.pixel_offst;

    meta_version = meta.meta_version;
    m_ncoords = meta.m_ncoords;
    nsrcs = meta.nsrcs;
    source_ids = meta.source_ids;
    kernel_size = meta.kernel_size;
  }

  void m_reset_buf() {}
};

EpicPixelTableMetaRows create_dummy_meta(int xdim, int ydim, int nsrcs = 1,
                                         int kernel_size = 5) {
  int ncoords = kernel_size * kernel_size * nsrcs;
  EpicPixelTableMetaRows meta(ncoords, nsrcs);

  srand(time(NULL));

  meta.pixel_coords.insert(meta.pixel_coords.end(), ncoords,
                           std::pair<int, int>(32, 33));
  meta.pixel_lm.insert(meta.pixel_lm.end(), ncoords, std::pair<int, int>(1, 1));
  meta.pixel_offst.insert(meta.pixel_offst.end(), ncoords,
                          std::pair<int, int>(0, 0));
  meta.source_ids.insert(meta.source_ids.end(), ncoords, 1);
  unsigned int _seed;
  meta.meta_version = rand_r(&_seed);
  meta.TransformPixCoords(xdim, ydim);

  return meta;
}

#endif  // SRC_EX_ORM_TYPES_HPP_
