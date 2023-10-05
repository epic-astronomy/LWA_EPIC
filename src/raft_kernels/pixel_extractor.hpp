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

#ifndef SRC_RAFT_KERNELS_PIXEL_EXTRACTOR_HPP_
#define SRC_RAFT_KERNELS_PIXEL_EXTRACTOR_HPP_
#include <glog/logging.h>

#include <chrono>
#include <cmath>
#include <memory>
#include <raft>
#include <raftio>
#include <variant>

#include "../ex/buffer.hpp"
#include "../ex/constants.h"
#include "../ex/metrics.hpp"
#include "../ex/orm_types.hpp"
#include "../ex/py_funcs.hpp"
#include "../ex/tensor.hpp"
#include "../ex/types.hpp"

template <typename _PldIn, typename _PldOut, class BufferMngr,
          typename BufConfig>
class PixelExtractor : public raft::kernel {
 private:
  /// Metadata object to store pixel coords that will be extracted from the
  /// image
  EpicPixelTableMetaRows m_pixmeta_rows;
  EpicPixelTableMetaRows _dummy_meta;
  std::unique_ptr<BufferMngr> m_buf_mngr{nullptr};
  bool is_meta_initialized{false};
  constexpr static size_t m_nbufs{20};
  constexpr static size_t m_maxiters{5};
  PSTensor<float> m_img_tensor;
  size_t m_xdim;
  size_t m_ydim;
  size_t m_nchan;

  unsigned int m_rt_gauge_id{0};
  Timer m_timer;

 public:
  PixelExtractor(BufConfig p_config,
                 const EpicPixelTableMetaRows& p_init_metarows, size_t p_xdim,
                 size_t p_ydim, size_t p_inchan)
      : raft::kernel(),
        m_pixmeta_rows(p_init_metarows),
        m_img_tensor(p_inchan, p_xdim, p_ydim),
        m_xdim(p_xdim),
        m_ydim(p_ydim),
        m_nchan(p_inchan) {
    input.addPort<EpicPixelTableMetaRows>("meta_pixel_rows");
    input.addPort<_PldIn>("in_img");

    output.addPort<_PldOut>("out_pix_rows");
    output.addPort<_PldIn>("out_img");

    m_buf_mngr.reset();
    m_buf_mngr = std::make_unique<BufferMngr>(m_nbufs, m_maxiters, p_config);
    for (size_t i = 0; i < m_nbufs; ++i) {
      _PldOut pld = m_buf_mngr.get()->acquire_buf();
      if (!pld) {
        LOG(FATAL) << "Null payload. Unable to allocate memory for the pixel "
                      "data buffer";
      }
      pld.get_mbuf()->copy_meta(m_pixmeta_rows);
    }

    m_rt_gauge_id = PrometheusExporter::AddRuntimeSummaryLabel(
        {{"type", "exec_time"},
         {"kernel", "pixel_extractor"},
         {"units", "s"},
         {"kernel_id", std::to_string(this->get_id())}});
  }

  raft::kstatus run() override {
    m_timer.Tick();
    // check if there are updates to the pixel meta rows
    if (input["meta_pixel_rows"].size() > 0) {
      input["meta_pixel_rows"].pop(_dummy_meta);
      if (_dummy_meta.meta_version != -1 &&
          _dummy_meta.meta_version != m_pixmeta_rows.meta_version) {
        m_pixmeta_rows = _dummy_meta;
      }
    }

    if (input["in_img"].size() == 0) {
      return raft::proceed;
    }

    _PldIn in_img;
    input["in_img"].pop(in_img);

    if (m_pixmeta_rows.meta_version == -1) {
      // the indices aren't there yet
      // unlikely to happen
      output["out_img"].push(in_img);
      return raft::proceed;
    }

    _PldOut out_pix_rows = m_buf_mngr.get()->acquire_buf();
    // this takes care of resizing the data and meta vectors
    out_pix_rows.get_mbuf()->copy_meta(m_pixmeta_rows);

    m_img_tensor.assign_data(in_img.get_mbuf()->GetDataPtr());

    // copy the image metadata
    out_pix_rows.get_mbuf()->m_img_metadata =
        in_img.get_mbuf()->GetMetadataRef();
    out_pix_rows.get_mbuf()->m_uuid = GetRandomUuid();

    // Extract pixels into the output payload
    VLOG(2) << "Extracting pixels into the payload";
    VLOG(2) << "Ncoords: " << m_pixmeta_rows.pixel_coords_sft.size() << " "
            << out_pix_rows.get_mbuf()->m_nchan;
    m_img_tensor.extract_pixels(m_pixmeta_rows,
                                out_pix_rows.get_mbuf()->pixel_values.get());

    output["out_pix_rows"].push(out_pix_rows);
    output["out_img"].push(in_img);

    m_timer.Tock();
    PrometheusExporter::ObserveRunTimeValue(m_rt_gauge_id, m_timer.Duration());
    return raft::proceed;
  }
};

#endif  // SRC_RAFT_KERNELS_PIXEL_EXTRACTOR_HPP_
