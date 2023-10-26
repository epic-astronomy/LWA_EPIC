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

#ifndef SRC_RAFT_KERNELS_CORRELATOR_HPP_
#define SRC_RAFT_KERNELS_CORRELATOR_HPP_
#include <glog/logging.h>

#include <chrono>
#include <cmath>
#include <memory>
#include <raft>
#include <raftio>
#include <utility>
#include <variant>

#include "../ex/constants.h"
#include "../ex/metrics.hpp"
#include "../ex/types.hpp"

template <class _Payload, class _Correlator>
class CorrelatorRft : public raft::kernel {
 private:
  std::unique_ptr<_Correlator> m_correlator{nullptr};
  int m_nchan{0};
  uint64_t m_chan0{0};
  int m_ngulps_per_img{1};
  int m_gulp_counter{1};
  bool m_is_first{false};
  bool m_is_last{false};
  int m_grid_size{0};
  float m_grid_res{0};
  int m_npols{0};
  int m_support{0};
  uint64_t m_seq_start_id{0};
  float m_delta{1};
  unsigned int m_rt_gauge_id{0};

 public:
  explicit CorrelatorRft(std::unique_ptr<_Correlator>* p_correlator)
      : raft::kernel(), m_correlator(std::move(*p_correlator)) {
    m_ngulps_per_img = m_correlator.get()->GetNumGulpsPerImg();
    m_grid_res = m_correlator.get()->GetGridRes();
    m_grid_size = m_correlator.get()->GetGridSize();
    m_npols = m_correlator.get()->GetNumPols();
    m_support = m_correlator.get()->GetSupportSize();
    m_delta = m_correlator.get()->GetScalingLen();
    input.addPort<_Payload>("gulp");
    // using out_t = typ
    output.addPort<typename _Correlator::payload_t>("img");
    output.addPort<typename _Correlator::payload_t>("img_stream");

    m_rt_gauge_id = PrometheusExporter::AddRuntimeSummaryLabel(
        {{"type", "info"},
         {"data", "channel"},
         {"units", "channel_number"},
         {"kernel_id", std::to_string(this->get_id())}});
  }

  raft::kstatus run() override {
    VLOG(2) << "Inside correlator rft";
    _Payload pld;
    input["gulp"].pop(pld);
    VLOG(2) << "Payload received";

    if (!pld) {
      return raft::proceed;
    }

    auto& gulp_metadata = pld.get_mbuf()->GetMetadataRef();
    VLOG(2) << "Acquiring metadata";
    for (auto it = gulp_metadata.begin(); it != gulp_metadata.end(); ++it) {
      VLOG(3) << "Key: " << it->first;
    }
    auto nchan = std::get<uint8_t>(gulp_metadata["nchan"]);
    auto chan0 = std::get<int64_t>(gulp_metadata["chan0"]);

    VLOG(2) << "nchan: " << int(nchan) << " chan0: " << chan0;
    PrometheusExporter::ObserveRunTimeValue(m_rt_gauge_id, chan0);

    // initialization or change in the spectral window
    if (m_correlator.get()->ResetImagingConfig(nchan, chan0)) {
      m_delta = m_correlator.get()->GetScalingLen();
      m_gulp_counter = 1;
    }

    m_is_first = m_gulp_counter == 1 ? true : false;
    m_is_last = m_gulp_counter == m_ngulps_per_img ? true : false;

    VLOG(2) << "Setting the start id";
    if (m_is_first) {
      m_seq_start_id = std::get<uint64_t>(gulp_metadata["seq_start"]);
    }

    if (m_is_last) {
      VLOG(3) << "Last gulp. Preparing metadata";
      // prepare the metadata for the image
      auto buf = m_correlator.get()->GetEmptyBuf();
      CHECK(static_cast<bool>(buf)) << "Correlator buffer allocation failed";
      auto& img_metadata = buf.get_mbuf()->GetMetadataRef();
      img_metadata = gulp_metadata;  // pld.get_mbuf()->GetMetadataRef();
      img_metadata["seq_start"] = m_seq_start_id;
      img_metadata["nseqs"] =
          std::get<int>(img_metadata["nseqs"]) * m_ngulps_per_img;
      img_metadata["img_len_ms"] =
          std::get<double>(img_metadata["gulp_len_ms"]) * m_ngulps_per_img;
      img_metadata["grid_size"] = m_grid_size;
      img_metadata["grid_res"] = m_grid_res;
      img_metadata["npols"] = m_npols;
      img_metadata["support_size"] = m_support;
      img_metadata["nchan"] = nchan;
      img_metadata["chan0"] = chan0;
      // img_metadata["cfreq"] = int((chan0+ceil(nchan/2f))*BANDWIDTH);
      VLOG(3)
          << "Processing gulp at: "
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::high_resolution_clock::now().time_since_epoch())
                 .count();
      m_correlator.get()->ProcessGulp(
          pld.get_mbuf()->GetDataPtr(), buf.get_mbuf()->GetDataPtr(),
          m_is_first, m_is_last, static_cast<int>(chan0), m_delta);

      m_gulp_counter = 1;
      VLOG(3) << "Pushing the output image";
      output["img"].push(buf);
      output["img_stream"].push(buf);

      return raft::proceed;
    }

    // If it's anything other than the final gulp, proceed right away. The
    // ProcessGulp function images the data in streams with asynchronous
    // reads/writes. Hence the next gulp won't have to wait for the current one
    // to complete thereby keeping the GPU completely occupied.

    m_correlator.get()->ProcessGulp(pld.get_mbuf()->GetDataPtr(),
                                     static_cast<float*>(nullptr), m_is_first,
                                     m_is_last);
    ++m_gulp_counter;
    return raft::proceed;
  }
};

#endif  // SRC_RAFT_KERNELS_CORRELATOR_HPP_
