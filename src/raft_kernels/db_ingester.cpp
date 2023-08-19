#ifndef DB_INGESTER
#define DB_INGESTER
#include "../ex/buffer.hpp"
#include "../ex/constants.h"
#include "../ex/db_helpers.hpp"
#include "../ex/orm_types.hpp"
#include "../ex/py_funcs.hpp"
#include "../ex/tensor.hpp"
#include "../ex/types.hpp"
#include <chrono>
#include <cmath>
#include <glog/logging.h>
#include <memory>
#include <raft>
#include <raftio>
#include <variant>

template<typename _PldIn>
class DBIngester_rft : public raft::kernel
{ // extraction kernel size
    int m_ext_kernel_size{ 5 };
    int m_nkernel_elems{ 25 };
    bool is_db_alive{ false };
    std::unique_ptr<pqxx::connection> m_pg_conn;
    std::unique_ptr<pqxx::work> m_db_T;
    const std::string m_pix_stmnt_id{ "insert_pixels" };
    const std::string m_meta_stmnt_id{ "insert_meta" };
    std::string m_pix_stmnt_id_n{ m_pix_stmnt_id + "_1" };
    std::string m_meta_stmnt_id_n{ m_meta_stmnt_id + "_1" };
    std::map<int, std::pair<std::string, std::string>> m_avail_ksizes;

  public:
    // https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING
    DBIngester_rft(int p_kernel_size = 1, std::string p_conn_string = "dbname=epic")
      : raft::kernel()
    {
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

    bool is_update_stmnt_ids(int p_kernel_size){
      if(m_avail_ksizes.count(p_kernel_size)==1){
        auto ids = m_avail_ksizes.at(p_kernel_size);
        m_pix_stmnt_id_n = ids.first;
        m_meta_stmnt_id_n = ids.second;
        return false;
      }
      m_pix_stmnt_id_n = m_pix_stmnt_id + "_" + std::to_string(p_kernel_size);
      m_meta_stmnt_id_n = m_meta_stmnt_id + "_" + std::to_string(p_kernel_size);
      m_avail_ksizes[p_kernel_size] = std::pair(m_pix_stmnt_id_n, m_meta_stmnt_id_n);

      return true;
    }

    void update_prepared_stmnts(int p_kernel_size)
    {
        if (p_kernel_size == m_ext_kernel_size ) {
            return;
        }

        m_ext_kernel_size = p_kernel_size;
        m_nkernel_elems = p_kernel_size * p_kernel_size;

        if(is_update_stmnt_ids(p_kernel_size)){
          m_pg_conn.get()->prepare(
            m_pix_stmnt_id_n, get_pixel_insert_stmnt_n(m_nkernel_elems));

          m_pg_conn.get()->prepare(
            m_meta_stmnt_id_n, get_img_meta_insert_stmnt_n(1)); // one row per image
        }
    }

    virtual raft::kstatus run() override
    {
        _PldIn pld;
        input["in_pixel_rows"].pop(pld);
        update_prepared_stmnts(pld.get_mbuf()->kernel_size);
        try {
            ingest_payload(
              pld,
              *m_db_T,
              m_nkernel_elems,
              m_pix_stmnt_id_n,
              m_meta_stmnt_id_n);
            m_db_T.get()->commit();
        } catch (const std::exception& e) {
            LOG(ERROR) << e.what();
        }
        return raft::proceed;
    }
};

#endif /* DB_INGESTER */
