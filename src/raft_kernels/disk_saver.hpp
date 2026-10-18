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
 * stops because of the long-integration archive. Write errors are logged
 * (rate-limited) and the accumulation is dropped.
 *
 * The registration goes through a PgLink: it is written off the writer's
 * thread, kept while the node's database is unavailable and dropped (with
 * rate-limited logs) when too many wait. A database error never stops the
 * imager.
 */
template <class Payload>
class DiskSaverRft : public raft::kernel
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
  std::unique_ptr<PgLink> m_db;
  // last: constructed after, and destroyed before, everything the writer uses
  std::unique_ptr<BoundedWorker<Job>> m_writer;

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
    auto meta = std::make_shared<dict_t>(std::move(job.meta));
    auto stmt_id = m_file_stmt_id;
    m_db->Submit("file " + filename, {{m_file_stmt_id, m_db_insert_stmt}},
                 [meta, stmt_id](pqxx::work& tx) {
                   InsertFileMetaToDb(*meta, &tx, stmt_id);
                 });
  }

 public:
  explicit DiskSaverRft(std::string p_db_schema = "public",
                        std::string p_out_dir = "./",
                        std::string p_db_conn_str = "", size_t p_queue_len = 4)
      : raft::kernel() /* , EpicLiveStream() */ {
    input.addPort<Payload>("image");
    m_filename_db_schema = p_db_schema != "" ? p_db_schema : "public";
    m_db_insert_stmt = GetFileMetaInsertStmt(m_filename_db_schema);
    m_out_dir = p_out_dir;
    // a few files a minute: 256 cover hours of an unavailable database
    m_db = std::make_unique<PgLink>("disk saver", p_db_conn_str, 256);

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
