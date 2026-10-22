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

#ifndef SRC_RAFT_KERNELS_CHAN_REDUCER_HPP_
#define SRC_RAFT_KERNELS_CHAN_REDUCER_HPP_

#include <chrono>
#include <cmath>
#include <algorithm>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <variant>

#include "../ex/buffer.hpp"
#include "../ex/constants.h"
#include "../ex/metrics.hpp"
#include "../ex/py_funcs.hpp"
#include "../ex/sky_publisher.hpp"
#include "../ex/tensor.hpp"
#include "../ex/types.hpp"
#include "glog/logging.h"
#include "raft"
#include "raftio"

/**
 * @brief Raft kernel bin the channels in the input image
 *
 * @tparam _PldIn Input payload type
 * @tparam BufferMngr Buffer manager type
 * @tparam _PldOut Output payload type
 */
template <typename _PldIn, class BufferMngr, typename _PldOut = _PldIn>
class ChanReducerRft : public raft::kernel {
 private:
  /// @brief Number of channels to combine
  const int m_ncombine{4};
  float m_norm{1};
  bool m_is_chan_avg{false};
  std::unique_ptr<BufferMngr> m_buf_mngr{nullptr};
  // images held for the downstream kernels: 300 at 128 x 128 (12 s at 24
  // images/s), fewer for larger images so the pool stays near 5 GB
  size_t m_nbufs{300};
  static constexpr size_t m_max_buf_reqs{10};
  size_t m_xdim{128};
  size_t m_ydim{128};
  size_t m_in_nchan{128};
  size_t m_out_nchan{32};
  PSTensor<float> m_in_tensor;
  PSTensor<float> m_out_tensor;
  static constexpr int NSTOKES{4};
  unsigned int m_rt_gauge_id{0};
  Timer m_timer;
  std::set<int> m_hc_chans;
  /// Input image being published (keeps its buffer until the worker has read it).
  /// Declared before m_sky so the worker stops before the buffer is released.
  std::optional<_PldIn> m_sky_hold;
  /// Stokes I images for the all-sky monitor, published on their own thread
  /// (null when --sky_shm is empty)
  std::unique_ptr<SkyPublishWorker> m_sky{nullptr};
  uint64_t m_sky_dropped_logged{0};
  std::chrono::steady_clock::time_point m_sky_last_log{std::chrono::steady_clock::now()};

  template <typename M>
  static double MetaNum(M& p_meta, const char* p_key) {
    auto it = p_meta.find(p_key);
    if (it == p_meta.end()) return 0;
    return std::visit(
        [](auto&& v) -> double {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_arithmetic_v<T>) return static_cast<double>(v);
          return 0.0;
        },
        it->second);
  }

  template <typename M>
  static uint64_t MetaU64(M& p_meta, const char* p_key) {
    auto it = p_meta.find(p_key);
    if (it == p_meta.end()) return 0;
    return std::visit(
        [](auto&& v) -> uint64_t {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_integral_v<T>) return static_cast<uint64_t>(v);
          if constexpr (std::is_floating_point_v<T>) return static_cast<uint64_t>(v);
          return 0;
        },
        it->second);
  }

  /// Hand the full-resolution image to the sky ring's writer thread. Never
  /// waits: while the previous image is still being written this one is
  /// dropped for the monitor (and counted), not for the imager.
  void PublishSky(_PldIn& p_pld) {
    if (m_sky->Failed()) {
      LOG(ERROR) << "Sky publisher disabled: " << m_sky->Error();
      m_sky.reset();
      m_sky_hold.reset();
      return;
    }
    if (m_sky_hold && !m_sky->InFlight()) {
      m_sky_hold.reset();  // the worker has read it: release the buffer here
    }
    if (!m_sky_hold) {
      auto& meta = p_pld.get_mbuf()->GetMetadataRef();
      SkyPublisher::ImageInfo info;
      info.seq_start = MetaU64(meta, "seq_start");
      info.nseqs = static_cast<uint32_t>(MetaU64(meta, "nseqs"));
      info.chan0 = static_cast<uint32_t>(MetaU64(meta, "chan0"));
      info.grid_res = MetaNum(meta, "grid_res");
      info.img_len_ms = MetaNum(meta, "img_len_ms");
      info.gain_version = MetaU64(meta, "gain_version");  // set by the correlator with --gain_table
      if (const uint64_t npix = MetaU64(meta, "sky_npix"); npix > 0) {
        // regridded to the monitor's pixels on the GPU (FX imaging)
        info.layout = 1;
        info.npix = static_cast<uint32_t>(npix);
        info.table_version = MetaU64(meta, "sky_table_version");
        info.pix = p_pld.get_mbuf()->GetDataPtr() + MetaU64(meta, "sky_pix_offset");
        info.peel_mask = static_cast<uint32_t>(MetaU64(meta, "sky_peel_mask"));
        info.peel_term = MetaU64(meta, "sky_peel_term") != 0;
      }
      m_sky_hold.emplace(p_pld);
      if (!m_sky->TrySubmit(p_pld.get_mbuf()->GetDataPtr(), static_cast<int>(m_in_nchan), info)) {
        m_sky_hold.reset();
      }
    } else {
      m_sky->CountDropped();  // still busy with the previous image
    }
    auto now = std::chrono::steady_clock::now();
    if (now - m_sky_last_log > std::chrono::seconds(60) && m_sky->Dropped() > m_sky_dropped_logged) {
      LOG(WARNING) << "Sky publisher: " << m_sky->Dropped() - m_sky_dropped_logged
                   << " images not published in the last minute (writer busy); "
                   << m_sky->Published() << " published in total, last took " << m_sky->LastMs() << " ms";
      m_sky_dropped_logged = m_sky->Dropped();
      m_sky_last_log = now;
    }
  }

 public:
  /**
   * @brief Construct a new ChanReducerRft object
   *
   * @param p_ncombine Channel binning factor
   * @param p_xdim X side of the image
   * @param p_ydim Y side of the image
   * @param p_in_nchan Number of input channels to the reducer
   */
  ChanReducerRft(int p_ncombine, int p_xdim, int p_ydim, int p_in_nchan,
                 const std::string& p_sky_shm = "", int p_sky_slots = 64,
                 int p_sky_nbin = 1)
      : raft::kernel(),
        m_ncombine(p_ncombine),
        m_xdim(p_xdim),
        m_ydim(p_ydim),
        m_in_nchan(p_in_nchan),
        m_in_tensor(PSTensor<float>(m_in_nchan, m_xdim, m_ydim)),
        m_out_tensor(
            PSTensor<float>(size_t(p_in_nchan / p_ncombine), m_xdim, m_ydim)),
        m_hc_chans(GetHealthCheckChans<LWA_SV>()) {
    input.addPort<_PldIn>("in_img");
    output.addPort<_PldOut>("out_img");
    output.addPort<uint64_t>("seq_start_id");

    if (m_in_nchan % m_ncombine != 0) {
      LOG(FATAL) << "The number of output channels: " << m_in_nchan
                 << " cannot be binned by a factor of " << m_ncombine << ". ";
    }

    m_out_nchan = m_in_nchan / m_ncombine;
    // none without combining: the images go on as they are
    m_nbufs = m_ncombine == 1 ? 1 : std::max<size_t>(64, 300 * (128 * 128) / std::max<size_t>(1, m_xdim * m_ydim));
    m_buf_mngr.reset(new BufferMngr(m_nbufs,
                                    m_xdim * m_ydim * m_out_nchan * NSTOKES,
                                    m_max_buf_reqs, false));

    if (!p_sky_shm.empty()) {
      try {
        m_sky = std::make_unique<SkyPublishWorker>(std::make_unique<SkyPublisher>(
            p_sky_shm, p_sky_slots, static_cast<int>(m_in_nchan), static_cast<int>(m_xdim),
            p_sky_nbin));
        LOG(INFO) << "Publishing Stokes I images to " << m_sky->Path();
      } catch (const std::exception& e) {
        LOG(ERROR) << "Sky publisher not started: " << e.what();
      }
    }

    m_rt_gauge_id = PrometheusExporter::AddRuntimeSummaryLabel(
        {{"type", "exec_time"},
         {"kernel", "chan_reducer"},
         {"units", "seconds"},
         {"kernel_id", std::to_string(this->get_id())}});
  }

  raft::kstatus run() override {
    m_timer.Tick();
    _PldIn in_pld;
    raft::signal in_sig=raft::none;
    input["in_img"].pop(in_pld, &in_sig);
    if(!static_cast<bool>(in_pld) || in_sig==INVALID_FREQ){
      LOG_EVERY_N(WARNING, 3000) << "Health Checks. Ignoring data";
      // in_pld = _PldIn();  // this releases the lock without waiting for raft's
                          // runtime to destroy the object
      m_timer.Tock();
      PrometheusExporter::ObserveRunTimeValue(m_rt_gauge_id, m_timer.Duration());
      return raft::proceed;
    }
    //auto chan0 =
        //std::get<int64_t>(in_pld.get_mbuf()->GetMetadataRef()["chan0"]);
    // if (m_hc_chans.count(chan0) > 0) {
    //   LOG_EVERY_N(WARNING, 3000) << "Health Checks. Ignoring data";
    //   // in_pld = _PldIn();  // this releases the lock without waiting for raft's
    //                       // runtime to destroy the object
    //   return raft::proceed;
    // }

    if (m_sky) PublishSky(in_pld);

    if (m_ncombine == 1) {
      // nothing to combine: the image itself goes on
      auto& meta = in_pld.get_mbuf()->GetMetadataRef();
      meta["chan_width"] = static_cast<int>(std::lround(BANDWIDTH));
      output["out_img"].push(in_pld);
      output["seq_start_id"].push(std::get<uint64_t>(meta["seq_start"]));
      m_timer.Tock();
      PrometheusExporter::ObserveRunTimeValue(m_rt_gauge_id, m_timer.Duration());
      return raft::proceed;
    }

    auto out_pld = m_buf_mngr->acquire_buf();
    if (!out_pld) {
      LOG(FATAL) << "Memory buffer full in ChanReducer";
    }

    auto& out_meta = out_pld.get_mbuf()->GetMetadataRef();

    out_meta = in_pld.get_mbuf()->GetMetadataRef();
    out_meta["nchan"] = (uint8_t)m_out_nchan;
    // stored as an int (Hz). The NDP channel width is not an integer.
    out_meta["chan_width"] = static_cast<int>(std::lround(BANDWIDTH * m_ncombine));

    m_in_tensor.assign_data(in_pld.get_mbuf()->GetDataPtr());
    m_out_tensor.assign_data(out_pld.get_mbuf()->GetDataPtr());

    m_in_tensor.combine_channels(&m_out_tensor);

    output["out_img"].push(out_pld);
    output["seq_start_id"].push(std::get<uint64_t>(out_meta["seq_start"]));

    m_timer.Tock();
    PrometheusExporter::ObserveRunTimeValue(m_rt_gauge_id, m_timer.Duration());
    return raft::proceed;
  }
};

#endif  // SRC_RAFT_KERNELS_CHAN_REDUCER_HPP_
