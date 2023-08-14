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

  public:
    // https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING
    DBIngester_rft(int p_kernel_size = 1, std::string p_conn_string = "dbname=epic")
      : raft::kernel()
    {
        m_ext_kernel_size = p_kernel_size;
        m_nkernel_elems = p_kernel_size * p_kernel_size;
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

        m_pg_conn.get()->prepare(
          m_pix_stmnt_id, get_pixel_insert_stmnt_n(m_nkernel_elems));

        m_pg_conn.get()->prepare(
          m_meta_stmnt_id, get_img_meta_insert_stmnt_n(m_nkernel_elems));
    }

    virtual raft::kstatus run() override
    {
        _PldIn pld;
        input["in_pixel_rows"].pop(pld);
        try {
            ingest_payload(
              pld,
              *m_db_T,
              m_nkernel_elems,
              m_pix_stmnt_id,
              m_meta_stmnt_id);
            m_db_T.get()->commit();
        } catch (const std::exception& e) {
            LOG(ERROR) << e.what();
        }
        return raft::proceed;
    }
};


#endif /* DB_INGESTER */
