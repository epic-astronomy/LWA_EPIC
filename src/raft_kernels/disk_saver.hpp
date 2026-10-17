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

#ifndef SRC_RAFT_KERNELS_DISK_SAVER_HPP_
#define SRC_RAFT_KERNELS_DISK_SAVER_HPP_
#include <glog/logging.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <raft>
#include <raftio>
#include <string>
#include <vector>

#include "../ex/async_writer.hpp"
#include "../ex/constants.h"
#include "../ex/db_helpers.hpp"
#include "../ex/metrics.hpp"
#include "../ex/py_funcs.hpp"
#include "../ex/types.hpp"
#include "../ex/video_streaming.hpp"
// #include "./epic_live_streamer.hpp"

/**
 * @brief Writes the long accumulations (FITS, quick look, daily digest) and
 * registers each file in epic_files_metadata2.
 *
 * The writing happens off the pipeline: run() copies the accumulation into a
 * bounded queue (m_queue_len entries) served by a writer thread and returns at
 * once. A full queue (the writer stalled, e.g. on a hung NFS mount) drops the
 * accumulation with a rate-limited warning, so the imager never stalls or
 * stops because of the long-integration archive. Write and database errors
 * are logged (rate-limited) and the accumulation is dropped.
 */
template <class Payload>
class DiskSaverRft : public raft::kernel,
                     protected PgDbConnectMixin
/* ,protected EpicLiveStream */ {
 private:
  struct Job {
    std::vector<float> data;
    dict_t meta;
    int imsize{0};
    int nchan{0};
  };

  unsigned int m_rt_gauge_id{0};
  std::string m_db_insert_stmt;
  Timer m_timer;

  std::string m_file_stmt_id{"insert_file_meta"};
  std::string m_filename_db_schema{"public"};
  std::string m_out_dir{"./"};

  RateLimit m_drop_log{std::chrono::seconds(300)};
  RateLimit m_write_log{std::chrono::seconds(300)};
  RateLimit m_db_log{std::chrono::seconds(300)};
  std::chrono::steady_clock::time_point m_next_reconnect{};
  bool m_db_ok{true};
  // last: constructed after, and destroyed before, everything the writer uses
  std::unique_ptr<BoundedWorker<Job>> m_writer;

  /// Reconnect after a database failure, at most once a minute.
  bool EnsureDb() {
    if (m_db_ok) return true;
    auto now = std::chrono::steady_clock::now();
    if (now < m_next_reconnect) return false;
    m_next_reconnect = now + std::chrono::seconds(60);
    try {
      m_pg_conn = std::make_unique<pqxx::connection>(m_conn_str);
      m_pg_conn->prepare(m_file_stmt_id, m_db_insert_stmt);
      m_db_ok = true;
      LOG(INFO) << "disk saver: database connection restored";
    } catch (const std::exception& e) {
      size_t held;
      if (m_db_log.Due(&held)) {
        LOG(WARNING) << "disk saver: database unavailable: " << e.what();
      }
    }
    return m_db_ok;
  }

  void Write(Job& job) {
    auto filename = SaveImageToDisk(job.imsize, job.nchan, job.data.data(),
                                    m_out_dir + "/", job.meta);
    if (filename.empty()) {
      size_t held;
      if (m_write_log.Due(&held)) {
        LOG(ERROR) << "disk saver: accumulation not written to " << m_out_dir
                   << " (" << held << " more since the last report)";
      }
      return;
    }
    job.meta["filename"] = filename;
    if (!EnsureDb()) return;
    try {
      pqxx::work tx(*m_pg_conn);
      InsertFileMetaToDb(job.meta, &tx, m_file_stmt_id);
      tx.commit();
    } catch (const std::exception& e) {
      m_db_ok = false;
      size_t held;
      if (m_db_log.Due(&held)) {
        LOG(ERROR) << "disk saver: " << filename
                   << " not registered in epic_files_metadata2: " << e.what()
                   << " (" << held << " more since the last report)";
      }
    }
  }

 public:
  explicit DiskSaverRft(std::string p_db_schema = "public",
                        std::string p_out_dir = "./",
                        std::string p_db_conn_str = "", size_t p_queue_len = 4)
      : raft::kernel(),
        PgDbConnectMixin(p_db_conn_str) /* , EpicLiveStream() */ {
    input.addPort<Payload>("image");
    m_filename_db_schema = p_db_schema != "" ? p_db_schema : "public";
    m_db_insert_stmt = GetFileMetaInsertStmt(m_filename_db_schema);
    m_out_dir = p_out_dir;
    LOG(INFO) << "Preparing stmt";
    this->prepare_stmt(m_file_stmt_id, m_db_insert_stmt);
    try {
      // a statement that hangs must not hold the writer forever
      pqxx::nontransaction(*m_pg_conn).exec("SET statement_timeout = 30000");
    } catch (const std::exception& e) {
      LOG(WARNING) << "disk saver: " << e.what();
    }

    m_rt_gauge_id = PrometheusExporter::AddRuntimeSummaryLabel(
        {{"type", "exec_time"},
         {"kernel", "disk_saver"},
         {"units", "s"},
         {"kernel_id", std::to_string(this->get_id())}});
    m_writer = std::make_unique<BoundedWorker<Job>>(
        p_queue_len, [this](Job& job) { this->Write(job); });
  }

  raft::kstatus run() override {
    m_timer.Tick();
    Payload pld;
    input["image"].pop(pld);

    if (!pld) {
      LOG(WARNING) << "Empty image received.";
      return raft::proceed;
    }

    Job job;
    job.meta = pld.get_mbuf()->GetMetadataRef();
    job.imsize = std::get<int>(job.meta["grid_size"]);
    job.nchan = std::get<uint8_t>(job.meta["nchan"]);
    int chan_width = std::get<int>(job.meta["chan_width"]);
    auto chan0 = std::get<int64_t>(job.meta["chan0"]);
    double cfreq = (chan0 + job.nchan * chan_width / BANDWIDTH * 0.5) *
                   BANDWIDTH * 1e-6;  // MHz
    auto time_tag = std::get<uint64_t>(job.meta["time_tag"]);
    job.meta["cfreq"] = cfreq;
    job.meta["epoch_time_s"] = time_tag / static_cast<double>(FS);
    // a copy, so the pipeline's buffer returns to its pool at once
    const float* src = pld.get_mbuf()->GetDataPtr();
    job.data.assign(src, src + size_t(job.imsize) * job.imsize * job.nchan * 4);
    pld = Payload();

    if (!m_writer->TryPush(std::move(job))) {
      size_t held;
      if (m_drop_log.Due(&held)) {
        LOG(WARNING) << "disk saver: writer busy (output directory slow or "
                        "hung?); accumulation dropped ("
                     << held << " more since the last report)";
      }
    }

    m_timer.Tock();
    PrometheusExporter::ObserveRunTimeValue(m_rt_gauge_id, m_timer.Duration());
    return raft::proceed;
  }
};

#endif  // SRC_RAFT_KERNELS_DISK_SAVER_HPP_
