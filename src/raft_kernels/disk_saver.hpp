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

#include "../ex/constants.h"
#include "../ex/db_helpers.hpp"
#include "../ex/metrics.hpp"
#include "../ex/py_funcs.hpp"
#include "../ex/types.hpp"

template <class Payload>
class DiskSaverRft : public raft::kernel, protected PgDbConnectMixin {
 private:
  std::string m_img_suffix;
  unsigned int m_rt_gauge_id{0};
  std::string m_db_insert_stmt;
  Timer m_timer;

  std::string m_file_stmt_id{"insert_file_meta"};

 public:
  explicit DiskSaverRft(std::string p_img_suffix = "0",
                        std::string p_db_conn_str = "dbname=epic")
      : raft::kernel(), PgDbConnectMixin(p_db_conn_str) {
    input.addPort<Payload>("image");
    m_db_insert_stmt = GetFileMetaInsertStmt();
    LOG(INFO) << "Preparing stmt";
    this->prepare_stmt(m_file_stmt_id, m_db_insert_stmt);

    m_img_suffix = p_img_suffix;
    m_rt_gauge_id = PrometheusExporter::AddRuntimeSummaryLabel(
        {{"type", "exec_time"},
         {"kernel", "disk_saver"},
         {"units", "s"},
         {"kernel_id", std::to_string(this->get_id())}});
  }

  raft::kstatus run() override {
    m_timer.Tick();
    using std::string_literals::operator""s;
    Payload pld;
    input["image"].pop(pld);

    if (!pld) {
      LOG(WARNING) << "Empty image received.";
      return raft::proceed;
    }

    auto& img_metadata = pld.get_mbuf()->GetMetadataRef();
    for (auto it = img_metadata.begin(); it != img_metadata.end(); ++it) {
      VLOG(3) << it->first << std::endl;
    }
    auto imsize = std::get<int>(img_metadata["grid_size"]);
    auto nchan = std::get<uint8_t>(img_metadata["nchan"]);
    int chan_width = std::get<int>(img_metadata["chan_width"]);
    auto chan0 = std::get<int64_t>(img_metadata["chan0"]);
    double cfreq = (chan0 + nchan * chan_width / BANDWIDTH * 0.5) * BANDWIDTH *
                   1e-6;  // MHz
    auto time_tag = std::get<uint64_t>(img_metadata["time_tag"]);
    // auto seq_start = std::get<uint64_t>(img_metadata["seq_start"]);
    double epoch_s = time_tag / static_cast<double>(FS);
    char _buf[50];
    std::snprintf(_buf, sizeof(_buf), "%.2fMHz_%.2f.fits", cfreq, epoch_s);
    std::string filename = "EPIC_" + std::string(_buf);
    img_metadata["filename"] = filename;
    img_metadata["cfreq"] = cfreq;
    img_metadata["epoch_time_s"] = epoch_s;
    SaveImageToDisk(imsize, nchan, pld.get_mbuf()->GetDataPtr(),
                    "test_image_"s + m_img_suffix, img_metadata);

    try {
      InsertFilenametoDb(&pld, this->m_db_T.get(), m_file_stmt_id);
      this->m_db_T.get()->commit();
    } catch (const std::exception& e) {
      LOG(FATAL) << e.what();
    }

    m_timer.Tock();
    PrometheusExporter::ObserveRunTimeValue(m_rt_gauge_id, m_timer.Duration());
    return raft::proceed;
  }
};

#endif  // SRC_RAFT_KERNELS_DISK_SAVER_HPP_
