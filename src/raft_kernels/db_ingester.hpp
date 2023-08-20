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
#include <glog/logging.h>

#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <raft>
#include <raftio>
#include <string>
#include <utility>
#include <variant>

#include "../ex/buffer.hpp"
#include "../ex/constants.h"
#include "../ex/db_helpers.hpp"
#include "../ex/orm_types.hpp"
#include "../ex/py_funcs.hpp"
#include "../ex/tensor.hpp"
#include "../ex/types.hpp"

template <typename _PldIn>
class DBIngester_rft : public raft::kernel {
  // extraction kernel size
  int m_ext_kernel_size{5};
  int m_nkernel_elems{25};
  bool is_db_alive{false};
  std::unique_ptr<pqxx::connection> m_pg_conn;
  std::unique_ptr<pqxx::work> m_db_T;
  const std::string m_pix_stmnt_id{"insert_pixels"};
  const std::string m_meta_stmnt_id{"insert_meta"};
  std::string m_pix_stmnt_id_n{m_pix_stmnt_id + "_1"};
  std::string m_meta_stmnt_id_n{m_meta_stmnt_id + "_1"};
  std::map<int, std::pair<std::string, std::string>> m_avail_ksizes;

 public:
  // https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING
  DBIngester_rft(int p_kernel_size = 1,
                 std::string p_conn_string = "dbname=epic")
      : raft::kernel() {
    input.addPort<_PldIn>("in_pixel_rows");

    try {
      m_pg_conn = std::make_unique<pqxx::connection>(p_conn_string);
    } catch (const std::exception& e) {
      LOG(FATAL) << e.what();
    }

    is_db_alive = true;
    m_db_T = std::make_unique<pqxx::work>(*(m_pg_conn.get()));

    // insertions will be made source-wise
    // for example, if the kernel size is 5 for a source, 25 rows will
    // be inserted in a single transaction
    update_prepared_stmnts(p_kernel_size);
  }

  bool is_update_stmnt_ids(int p_kernel_size) {
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

  void update_prepared_stmnts(int p_kernel_size) {
    if (p_kernel_size == m_ext_kernel_size) {
      return;
    }

    m_ext_kernel_size = p_kernel_size;
    m_nkernel_elems = p_kernel_size * p_kernel_size;

    if (is_update_stmnt_ids(p_kernel_size)) {
      m_pg_conn.get()->prepare(m_pix_stmnt_id_n,
                               get_pixel_insert_stmnt_n(m_nkernel_elems));

      m_pg_conn.get()->prepare(m_meta_stmnt_id_n, get_img_meta_insert_stmnt_n(
                                                      1));  // one row per image
    }
  }

  raft::kstatus run() override {
    _PldIn pld;
    input["in_pixel_rows"].pop(pld);
    update_prepared_stmnts(pld.get_mbuf()->kernel_size);
    try {
      ingest_payload(&pld, m_db_T.get(), m_nkernel_elems, m_pix_stmnt_id_n,
                     m_meta_stmnt_id_n);
      m_db_T.get()->commit();
    } catch (const std::exception& e) {
      LOG(ERROR) << e.what();
    }
    return raft::proceed;
  }
};
