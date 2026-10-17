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

#ifndef SRC_RAFT_KERNELS_DB_INGESTER_HPP_
#define SRC_RAFT_KERNELS_DB_INGESTER_HPP_
#include <glog/logging.h>

#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <pqxx/pqxx>
#include <raft>
#include <raftio>
#include <string>
#include <utility>
#include <variant>

#include "../ex/buffer.hpp"
#include "../ex/constants.h"
#include "../ex/db_helpers.hpp"
#include "../ex/metrics.hpp"
#include "../ex/orm_types.hpp"
#include "../ex/py_funcs.hpp"
#include "../ex/tensor.hpp"
#include "../ex/types.hpp"

/**
 * @brief Writes the extracted pixels (epic_pixels) and the sessions
 * (epic_img_metadata2) to the node's database.
 *
 * run() copies each image's rows out of the pipeline buffer and hands them to
 * a PgLink, whose writer thread inserts them, one transaction per image. The
 * pipeline never waits for the database: while it is unavailable the images
 * are kept (up to p_queue_len, about 100 s at 12 images/s), then dropped with
 * rate-limited logs; nothing here stops the imager. Counts of the waiting,
 * dropped and failed writes are exported as runtime gauges.
 */
template <typename _PldIn>
class DBIngesterRft : public raft::kernel {
 private:
  const std::string m_pix_stmnt_id{"insert_pixels"};
  const std::string m_meta_stmnt_id{"insert_meta"};
  std::string m_schema{"public"};

  unsigned int m_rt_gauge_id{0};
  unsigned int m_pending_gauge_id{0};
  unsigned int m_dropped_gauge_id{0};
  unsigned int m_failed_gauge_id{0};
  Timer m_timer;
  // last: destroyed first, while everything its writer uses still exists
  std::unique_ptr<PgLink> m_db;

 public:
  // https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING
  // https://www.postgresql.org/docs/current/libpq-envars.html
  DBIngesterRft(std::string p_schema = "public", int p_kernel_size = 1,
                std::string p_conn_string = "" /* EPIC_NODE_DSN or PG* env */,
                size_t p_queue_len = 1200)
      : raft::kernel() {
    input.addPort<_PldIn>("in_pixel_rows");
    (void)p_kernel_size;  // statements are prepared per kernel size on use
    m_schema = p_schema == "" ? "public" : p_schema;
    m_db = std::make_unique<PgLink>("db ingester", p_conn_string, p_queue_len);

    m_rt_gauge_id = PrometheusExporter::AddRuntimeSummaryLabel(
        {{"type", "exec_time"},
         {"kernel", "db_ingester"},
         {"units", "s"},
         {"kernel_id", std::to_string(this->get_id())}});
    auto gauge = [&](const char* what) {
      return PrometheusExporter::AddRunTimeGaugeLabel(
          {{"type", "db_writes"},
           {"kernel", "db_ingester"},
           {"value", what},
           {"kernel_id", std::to_string(this->get_id())}});
    };
    m_pending_gauge_id = gauge("pending");
    m_dropped_gauge_id = gauge("dropped");
    m_failed_gauge_id = gauge("failed");
  }

  raft::kstatus run() override {
    m_timer.Tick();
    _PldIn pld;
    input["in_pixel_rows"].pop(pld);
    if (pld.get_mbuf()->nsrcs == 0 || pld.get_mbuf()->meta_version == -1) {
      m_timer.Tock();
      return raft::proceed;
    }
    try {
      int kdim = pld.get_mbuf()->m_kernel_dim;
      int npix = kdim * kdim;
      auto job = MakePixelJob(&pld, npix);
      pld = _PldIn();  // the pipeline's buffer goes back to its pool now

      std::string pix_id = m_pix_stmnt_id + "_" + std::to_string(kdim);
      std::string meta_id = m_meta_stmnt_id + "_1";
      PgLink::Statements stmts{
          {pix_id, GetMultiPixelInsertStmnt(npix, m_schema)},
          {meta_id, GetMultiImgMetaInsertStmnt(1, m_schema)}};
      m_db->Submit("image " + job->time_stamp, std::move(stmts),
                   [job, pix_id, meta_id](pqxx::work& tx) {
                     IngestPixelJob(*job, &tx, pix_id, meta_id);
                   });
    } catch (const std::exception& e) {
      LOG(ERROR) << "db ingester: " << e.what();
    }
    PrometheusExporter::SetRunTimeGauge(m_pending_gauge_id, m_db->Pending());
    PrometheusExporter::SetRunTimeGauge(m_dropped_gauge_id, m_db->Dropped());
    PrometheusExporter::SetRunTimeGauge(m_failed_gauge_id, m_db->Failed());
    m_timer.Tock();
    PrometheusExporter::ObserveRunTimeValue(m_rt_gauge_id, m_timer.Duration());
    return raft::proceed;
  }
};

#endif  // SRC_RAFT_KERNELS_DB_INGESTER_HPP_
