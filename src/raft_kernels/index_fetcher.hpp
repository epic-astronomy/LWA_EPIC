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
  public:
    IndexFetcher_rft()
      : raft::kernel()
    {
        input.addPort<uint64_t>("tstart");
        output.addPort<EpicPixelTableMetaRows>("meta_pixel_rows");
    }

    virtual raft::kstatus run() override
    {
        uint64_t tstart;
        input["tstart"].pop(tstart);
        if(tstart==-1){
            //output["meta_pixel_rows"].push(EpicPixelTableMetaRows());
            return raft::proceed;
        }
        output["meta_pixel_rows"].push(create_dummy_meta(128, 128));
        LOG(INFO)<<"FIRED INDEX FETCHER";
        return raft::proceed;
    }
};
#endif /* INDEX_FETCHER */
