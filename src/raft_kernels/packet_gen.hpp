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

#ifndef SRC_RAFT_KERNELS_PACKET_GEN_HPP_
#define SRC_RAFT_KERNELS_PACKET_GEN_HPP_
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <memory>
#include <raft>
#include <raftio>
#include <utility>

#include "../ex/constants.h"
#include "../ex/metrics.hpp"
#include "../ex/types.hpp"

/**
 * @brief Raft kernel to generate gulps with support for timed and untimed
 * modes.
 *
 * @tparam PktAssmblr Packet assembler kernel
 */
template <class PktAssmblr>
class GulpGen_rft : public raft::kernel {
  using payload_t = typename PktAssmblr::payload_t;

 private:
  std::unique_ptr<PktAssmblr> m_assmblr;
  std::vector<double> m_get_ms, m_push_ms;
  /// Duration of the generator in seconds
  int m_timer;
  /// Flag to keep the program running forever
  bool m_perpetual{false};
  /// Flag to terminate the generator
  bool m_terminate{false};
  std::chrono::time_point<std::chrono::steady_clock> m_start_time;
  int m_num_empty_gulp{0};
  /// Seconds the pipeline gets to finish after a timed run before the process
  /// is ended (see run())
  const int m_exit_grace_s{30};
  /// Restart the imager if no valid gulp arrives for this long (seconds)
  const int m_max_no_data_s{300};
  std::chrono::time_point<std::chrono::steady_clock> m_last_valid_time;
  double m_gulp_duration;
  bool m_start_set{false};
  int m_rt_gauge_id{0};
  int _counter{0};
  // Timer m_timer;

 public:
  /**
   * @brief Construct a new GulpGen_rft object
   *
   * @param p_assmblr Pointer to the packet assembler
   * @param p_timer_s Total run time of the program in seconds. A negative value
   * will run it forever.
   *
   */
  GulpGen_rft(std::unique_ptr<PktAssmblr>& p_assmblr, int p_timer_s)
      : m_assmblr(std::move(p_assmblr)), m_timer(p_timer_s), raft::kernel() {
    if (m_timer < 0) m_perpetual = true;

    LOG_IF(FATAL, m_timer == 0)
        << "Run time for the gulp generator cannot be zero.";

    m_gulp_duration = m_assmblr.get()->GetNumSeqPerGulp() * SAMPLING_LEN_uS * 1e-6;

    if (m_timer > 0 && m_timer < m_gulp_duration) {
      LOG(WARNING) << "The specified runtime of " << m_timer
                   << " s is less than the gulp size (" << m_gulp_duration
                   << "). Adjusting the run time to " << m_gulp_duration
                   << " s.";
      m_timer = static_cast<int>(std::ceil(m_gulp_duration));
    }
    LOG(INFO) << "Initializing the Gulp generator.";
    LOG_IF(INFO, m_perpetual) << "Running in continuous mode.";
    LOG_IF(INFO, !m_perpetual)
        << "Running in timed mode for " << p_timer_s << " seconds.";

    output.addPort<payload_t>("gulp");

    m_rt_gauge_id = PrometheusExporter::AddRuntimeSummaryLabel(
        {{"type", "status"},
         {"kernel", "packet_gen"},
         {"units", "status"},
         {"kernel_id", std::to_string(this->get_id())}});
  }

  virtual raft::kstatus run() {
    if (!m_start_set) {
      m_start_time = std::chrono::steady_clock::now();
      m_last_valid_time = m_start_time;
      m_start_set = true;
    }
    while (m_perpetual || std::chrono::duration_cast<std::chrono::seconds>(
                              std::chrono::steady_clock::now() - m_start_time)
                                  .count() < m_timer) {
      VLOG(2) << "DURATION: "
              << std::chrono::duration_cast<std::chrono::seconds>(
                     std::chrono::steady_clock::now() - m_start_time)
                     .count();
      VLOG(3) << "Receiving the gulp";
      const auto t_get = std::chrono::steady_clock::now();
      auto gulp = m_assmblr.get()->get_gulp();
      m_get_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_get).count());
      VLOG(3) << "Received";
      if (!gulp) {
        // A gulp with too few packets. get_gulp() has already waited a whole
        // gulp on the network, so carry on at once: pausing here (as before,
        // ~0.4 s) lets the receive ring overflow and the following gulps come
        // back empty too. Restart only after a long stretch without any data.
        VLOG(2) << "Null gulp";
        ++m_num_empty_gulp;
        LOG_EVERY_N(INFO, 100) << "100 Empty gulps received";
        PrometheusExporter::ObserveRunTimeValue(m_rt_gauge_id, 0);
        const auto no_data_s = std::chrono::duration_cast<std::chrono::seconds>(
                                   std::chrono::steady_clock::now() - m_last_valid_time)
                                   .count();
        if (no_data_s > m_max_no_data_s) {
          LOG(FATAL) << "No valid gulp for " << no_data_s
                     << " s. Resetting the imager";
        }
        continue;
      }
      m_last_valid_time = std::chrono::steady_clock::now();
      auto& gulp_metadata = gulp.get_mbuf()->GetMetadataRef();
      auto chan0 = std::get<int64_t>(gulp_metadata["chan0"]);
      if(chan0<=0){
        LOG_EVERY_N(INFO, 100)<<"Gulp received with invalid frequency. Chan0: "<<chan0;
        continue;
      }
      // if (++_counter < 3000) {
      //   std::get<int64_t>(gulp.get_mbuf()->GetMetadataRef()["chan0"]);
      //   gulp.get_mbuf()->GetMetadataRef()["chan0"] = int64_t(440);
      // }
      const auto t_push = std::chrono::steady_clock::now();
      output["gulp"].push(gulp);
      m_push_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_push).count());
      if (m_push_ms.size() >= 240) {
        // how long assembling a gulp took, and handing it on (downstream back-pressure)
        auto stats = [](std::vector<double>& v) {
          std::sort(v.begin(), v.end());
          char b[96];
          std::snprintf(b, sizeof(b), "p50 %.2f p99 %.2f max %.2f ms", v[v.size() / 2], v[v.size() * 99 / 100], v.back());
          v.clear();
          return std::string(b);
        };
        LOG(INFO) << "Gulp generator: get " << stats(m_get_ms) << "; push " << stats(m_push_ms) << "; "
                  << m_assmblr.get()->TakeStats();
      }
      PrometheusExporter::ObserveRunTimeValue(m_rt_gauge_id, 1);

      if (m_terminate) break;
    }
    LOG(INFO) << "Stopping gulp gen";
    // RaftLib (Schedule::kernelRun) never finishes a kernel whose inputs are
    // empty: it returns before checking whether the producers have stopped,
    // so the downstream kernels poll their closed ports forever and a timed
    // run never exits. Give the pipeline time to push the last images
    // through, then end the process.
    std::thread([grace_s = m_exit_grace_s]() {
      std::this_thread::sleep_for(std::chrono::seconds(grace_s));
      LOG(WARNING) << "The pipeline did not finish " << grace_s
                   << " s after the gulp generator stopped. Exiting.";
      google::FlushLogFiles(google::GLOG_INFO);
      std::_Exit(0);
    }).detach();
    return raft::stop;
  }
};

#endif  // SRC_RAFT_KERNELS_PACKET_GEN_HPP_
