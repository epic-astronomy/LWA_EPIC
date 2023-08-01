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

template<typename _PldIn>
class DBIngester_rft : public raft::kernel
{
  public:
    DBIngester_rft()
      : raft::kernel()
    {
        input.addPort<_PldIn>("in_pixel_rows");
    }

    virtual raft::kstatus run() override
    {
        _PldIn pld;
        input["in_pixel_rows"].pop(pld);

        const auto* values = pld.get_mbuf()->pixel_values.get();

        int nchan = pld.get_mbuf()->m_nchan;

        for (int i = 0; i < nchan; ++i) {
            LOG(INFO) << i << " " << values[i];
        }

        return raft::proceed;
    }
};