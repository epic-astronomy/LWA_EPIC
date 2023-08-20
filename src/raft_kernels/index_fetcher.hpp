#ifndef INDEX_FETCHER
#define INDEX_FETCHER

#include "../ex/buffer.hpp"
#include "../ex/constants.h"
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

class IndexFetcher_rft : public raft::kernel
{
  protected:
    long int m_refresh_interval{ 10 };
    using high_res_tp = typename std::chrono::time_point<std::chrono::high_resolution_clock>;
    high_res_tp m_prev_refresh;
    // long int m_refresh_interval{ 10 };

    bool is_refresh_required()
    {
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - m_prev_refresh).count() >= m_refresh_interval) {
            m_prev_refresh = now;
            return true;
        }

        return false;
    }

    uint64_t m_tstart;

  public:
    IndexFetcher_rft(int p_refresh_interval = 10)
      : raft::kernel()
      , m_refresh_interval(p_refresh_interval)
    {
        input.addPort<uint64_t>("tstart");
        output.addPort<EpicPixelTableMetaRows>("meta_pixel_rows");
        m_prev_refresh = std::chrono::high_resolution_clock::now();
    }

    virtual raft::kstatus run() override
    {
        input["tstart"].pop(m_tstart);
        if (m_tstart == 0) {
            // output["meta_pixel_rows"].push(EpicPixelTableMetaRows());
            return raft::proceed;
        }
        if (is_refresh_required()) {
            output["meta_pixel_rows"].push(create_dummy_meta(128, 128));
            VLOG(2) << "FIRED INDEX FETCHER";
        }
        return raft::proceed;
    }
};
#endif /* INDEX_FETCHER */
