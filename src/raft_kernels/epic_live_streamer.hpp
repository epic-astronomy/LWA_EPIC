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

#ifndef SRC_RAFT_KERNELS_EPIC_LIVE_STREAMER_HPP_
#define SRC_RAFT_KERNELS_EPIC_LIVE_STREAMER_HPP_
#include <algorithm>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "../ex/constants.h"
#include "../ex/metrics.hpp"
#include "../ex/video_streaming.hpp"
#include "../ex/station_desc.hpp"
#include "glog/logging.h"
#include "raft"
#include "raftio"

template <typename _Pld>
class EpicLiveStream : public raft::kernel {
 private:
  Streamer* m_streamer{nullptr};
  bool m_is_streamer_set{false};
  Timer m_timer;
  int m_rt_gauge_id;
  std::set<int> m_hc_chans;  //  health check channel numbers

 public:
  EpicLiveStream() : raft::kernel(), m_hc_chans(GetHealthCheckChans<LWA_SV>()) {
    input.addPort<_Pld>("in_img");

    m_rt_gauge_id = PrometheusExporter::AddRuntimeSummaryLabel(
        {{"type", "exec_time"},
         {"kernel", "live_streamer"},
         {"units", "s"},
         {"kernel_id", std::to_string(this->get_id())}});
  }
  void SetStreamer(Streamer* p_streamer) {
    if (p_streamer != nullptr) {
      m_streamer = p_streamer;
      m_is_streamer_set = true;
    } else {
      LOG(FATAL) << "Pointer to the streamer is null";
    }
  }

  raft::kstatus run() override {
    m_timer.Tick();
    _Pld pld;
    input["in_img"].pop(pld);
    LOG_IF(FATAL, !m_is_streamer_set) << "Streamer is unintialzed";
    auto img_metadata = pld.get_mbuf()->GetMetadataRef();
    auto imsize = std::get<int>(img_metadata["grid_size"]);
    auto nchan = std::get<uint8_t>(img_metadata["nchan"]);
    int chan_width = std::get<int>(img_metadata["chan_width"]);
    auto chan0 = std::get<int64_t>(img_metadata["chan0"]);
    double cfreq = (chan0 + nchan * chan_width / BANDWIDTH * 0.5) * BANDWIDTH *
                   1e-6;  // MHz
    bool is_hc_freq = m_hc_chans.count(chan0) > 0 ? true : false;
    m_streamer->Stream(chan0, cfreq, pld.get_mbuf()->GetDataPtr(), is_hc_freq);
    m_timer.Tock();
    PrometheusExporter::ObserveRunTimeValue(m_rt_gauge_id, m_timer.Duration());
    return raft::proceed;
  }
};

#endif  // SRC_RAFT_KERNELS_EPIC_LIVE_STREAMER_HPP_
