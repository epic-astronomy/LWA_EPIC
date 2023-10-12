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

#include <glog/logging.h>
#include <hwy/aligned_allocator.h>

#include <algorithm>
#include <cassert>
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
  size_t m_kernel_size;
  size_t m_kernel_dim;
  std::vector<std::pair<int, int>> pixel_coords;
  std::vector<std::pair<int, int>> pixel_coords_sft;
  std::vector<std::pair<float, float>> pixel_lm;
  // std::vector<std::pair<float, float>> pixel_skypos;
  std::vector<std::pair<int, int>> pixel_offst;

  size_t m_ncoords;
  int meta_version{-1};

  EpicPixelTableMetaRows(int ncoords, int n_sources, int kernel_dim = 5) {
    nsrcs = n_sources;
    m_ncoords = ncoords;
    m_kernel_dim = kernel_dim;

    if (nsrcs == 0 || m_ncoords == 0) {
      return;
    }
    source_ids.resize(n_sources);
    pixel_coords.resize(ncoords);
    pixel_coords_sft.resize(ncoords);
    pixel_lm.resize(ncoords);
    // pixel_skypos.resize(ncoords);
    pixel_offst.resize(ncoords);

    assert(IsKernelSizeValid(n_sources, ncoords) &&
           "Invalid kernel size. Each source must have ksize * ksize number of "
           "sources.");

    m_kernel_size = ncoords / nsrcs;

    VLOG(3) << "Meta rows";
    VLOG(3) << "Nsrc: " << nsrcs << " Ncoords: " << m_ncoords
            << " Kernel size: " << m_kernel_size
            << " Kernel dim: " << m_kernel_dim;
  }

  EpicPixelTableMetaRows() {}

  /**
   * @brief Transform pixel indices into array indices
   *
   * The pixel indexing starts from the bottom left corner of the image, while
   * the array indexing starts from the top left corner. This function
   * performs the appropriate transformation between pixel and array indices.
   *
   * @param xdim X-dimension of the image
   * @param ydim Y-dimension of the image
   */
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
      pixel_coords_sft[i] = std::pair<int, int>(x, y);
    }
  }

  /**
   * @brief Compare this set of coords with another.
   *
   * Use this to check for any updates
   *
   * @param rhs Coordinate set to compare to
   * @return Whether or not the coordinate sets are identical
   */
  bool diff(const EpicPixelTableMetaRows& rhs) {
    return (m_ncoords != rhs.m_ncoords) || (pixel_coords != rhs.pixel_coords) ||
           (pixel_lm != rhs.pixel_lm) ||
           //    (pixel_skypos != rhs.pixel_skypos) ||
           (pixel_offst != rhs.pixel_offst);
  }

  /**
   * @brief Check if a square shaped kernel produces the current coordinate
   * set
   *
   * @param nsrcs Total number of sources
   * @param ncoords Total number of coordinates
   * @return Returns whether the kernel is square-shaped
   */
  bool IsKernelSizeValid(int nsrcs, int ncoords) {
    size_t nkernel_elems = ncoords / nsrcs;
    size_t ksize = sqrt(nkernel_elems);
    return nkernel_elems == ksize * ksize;
  }

  void print() {
    VLOG(3) << "Printing indices";
    VLOG(3) << pixel_coords.size() << " " << pixel_coords_sft.size() << " "
            << pixel_lm.size() << " " << pixel_offst.size();
    for (int i = 0; i < m_ncoords; ++i) {
      VLOG(3) << pixel_coords[i].first << " " << pixel_coords[i].second << " "
              << pixel_coords_sft[i].first << " " << pixel_coords_sft[i].second
              << " " << pixel_lm[i].first << " " << pixel_lm[i].second << " "
              << pixel_offst[i].first << " " << pixel_offst[i].second;
    }
  }
};

template <typename _Dtype>
struct EpicPixelTableDataRows : EpicPixelTableMetaRows {
  struct _config {
    int ncoords;
    int nsrcs;
    int nchan;
    int kernel_dim;
    bool CheckOpts() {
      bool flag = true;
      flag &= ncoords < 0 ? false : true;
      flag &= nchan < 0 ? false : true;
      flag &= nsrcs < 0 ? false : true;
      flag &= kernel_dim < 0 ? false : true;

      if (!flag) {
        VLOG(3) << ncoords << " " << nsrcs << " " << nchan << " " << kernel_dim;
      }

      return flag;
    }
  };

  using config_t = _config;
  size_t m_nchan{32};
  dict_t m_img_metadata;

  hwy::AlignedFreeUniquePtr<_Dtype[]> pixel_values;
  explicit EpicPixelTableDataRows(config_t config)
      : EpicPixelTableMetaRows(config.ncoords, config.nsrcs,
                               config.kernel_dim) {
    if (config.ncoords > 0) {
      pixel_values = std::move(hwy::AllocateAligned<_Dtype>(
          config.ncoords * NSTOKES * config.nchan));
    }
    m_nchan = config.nchan;
  }

  void copy_meta(const EpicPixelTableMetaRows& meta) {
    if (meta_version == meta.meta_version) {
      return;
    }
    VLOG(3) << "Copying metadata";

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
    m_kernel_size = meta.m_kernel_size;
    m_kernel_dim = meta.m_kernel_dim;
  }

  void ResetBuf() {}
};

EpicPixelTableMetaRows CreateDummyMeta(int xdim, int ydim, int nsrcs = 1,
                                       int kernel_dim = 5) {
  int ncoords = kernel_dim * kernel_dim * nsrcs;
  EpicPixelTableMetaRows meta(ncoords, nsrcs, kernel_dim);

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
