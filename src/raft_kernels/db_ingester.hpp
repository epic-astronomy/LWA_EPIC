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

template <typename _PldIn>
class DBIngesterRft : public raft::kernel {
 private:
  // extraction kernel size
  int m_ext_kernel_size{5};
  int m_nkernel_elems{25};
  bool is_db_alive{false};
  // Each image is written in its own transaction. A transaction opened at
  // start-up and reused would hold locks on epic_pixels and
  // epic_img_metadata2 until its first commit (hours, if no watched source is
  // up), blocking schema changes and partition maintenance.
  std::unique_ptr<pqxx::connection> m_pg_conn;
  std::string m_conn_string;
  const std::string m_pix_stmnt_id{"insert_pixels"};
  const std::string m_meta_stmnt_id{"insert_meta"};
  std::string m_pix_stmnt_id_n{m_pix_stmnt_id + "_1"};
  std::string m_meta_stmnt_id_n{m_meta_stmnt_id + "_1"};
  std::map<int, std::pair<std::string, std::string>> m_avail_ksizes;
  std::string m_schema{"public"};

  unsigned int m_rt_gauge_id{0};
  Timer m_timer;

 public:
  // https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING
  // https://www.postgresql.org/docs/current/libpq-envars.html
  DBIngesterRft(std::string p_schema="public",
                int p_kernel_size = 1,
                std::string p_conn_string = ""/* set from env variables*/
                )
      : raft::kernel() {
    input.addPort<_PldIn>("in_pixel_rows");

    m_conn_string = p_conn_string;
    try {
      m_pg_conn = std::make_unique<pqxx::connection>(p_conn_string);
    } catch (const std::exception& e) {
      LOG(FATAL) << e.what();
    }

    m_schema = p_schema==""?"public":p_schema;

    is_db_alive = true;

    // insertions will be made source-wise
    // for example, if the kernel size is 5 for a source, 25 rows will
    // be inserted in a single transaction
    UpdatePreparedStmnts(p_kernel_size);

    m_rt_gauge_id = PrometheusExporter::AddRuntimeSummaryLabel(
        {{"type", "exec_time"},
         {"kernel", "db_ingester"},
         {"units", "s"},
         {"kernel_id", std::to_string(this->get_id())}});
  }

  bool IsUpdateStmntIds(int p_kernel_size) {
    if (m_avail_ksizes.count(p_kernel_size) == 1) {
      auto ids = m_avail_ksizes.at(p_kernel_size);
      m_pix_stmnt_id_n = ids.first;
      m_meta_stmnt_id_n = ids.second;
      return false;
    }
    m_pix_stmnt_id_n = m_pix_stmnt_id + "_" + std::to_string(p_kernel_size);
    m_meta_stmnt_id_n = m_meta_stmnt_id + "_" + std::to_string(p_kernel_size);
    m_avail_ksizes[p_kernel_size] =
        std::pair(m_pix_stmnt_id_n, m_meta_stmnt_id_n);

    return true;
  }

  void UpdatePreparedStmnts(int p_kernel_size) {
    if (p_kernel_size == m_ext_kernel_size) {
      return;
    }

    m_ext_kernel_size = p_kernel_size;
    m_nkernel_elems = p_kernel_size * p_kernel_size;

    if (IsUpdateStmntIds(p_kernel_size)) {
      VLOG(3)<<"Preparing pixel insert stmnt";
      m_pg_conn.get()->prepare(m_pix_stmnt_id_n,
                               GetMultiPixelInsertStmnt(m_nkernel_elems, m_schema));

      VLOG(3)<<"Preparing meta insert statement";
      m_pg_conn.get()->prepare(m_meta_stmnt_id_n, GetMultiImgMetaInsertStmnt(
                                                      1, m_schema));  // one row per image

      VLOG(3)<<"Done preparing statements";
    }
  }

  void Reconnect() {
    try {
      m_pg_conn = std::make_unique<pqxx::connection>(m_conn_string);
      // prepared statements belong to the old connection
      m_avail_ksizes.clear();
      m_ext_kernel_size = -1;
    } catch (const std::exception& e) {
      LOG(ERROR) << "Reconnecting to the database failed: " << e.what();
    }
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
      if (!m_pg_conn->is_open()) {
        Reconnect();
      }
      UpdatePreparedStmnts(pld.get_mbuf()->m_kernel_dim);
      pqxx::work tx(*(m_pg_conn.get()));
      IngestPayload(&pld, &tx, m_nkernel_elems, m_pix_stmnt_id_n,
                    m_meta_stmnt_id_n);
      tx.commit();
    } catch (const pqxx::broken_connection& e) {
      LOG(ERROR) << "Lost the database connection: " << e.what();
      Reconnect();
    } catch (const std::exception& e) {
      LOG(ERROR) << e.what();
    }
    m_timer.Tock();
    PrometheusExporter::ObserveRunTimeValue(m_rt_gauge_id, m_timer.Duration());
    return raft::proceed;
  }
};

#endif  // SRC_RAFT_KERNELS_DB_INGESTER_HPP_
