#include "../ex/constants.h"
#include "../ex/types.hpp"
#include <chrono>
#include <glog/logging.h>
#include <memory>
#include <raft>
#include <raftio>

template<class Payload>
class dummy : public raft::kernel
{
  public:
    dummy()
      : raft::kernel()
    {
        input.addPort<Payload>("gulp");
    }

    virtual raft::kstatus run()
    {
        for (int i = 0; i < 3; ++i) {
            if(input["gulp"].size()==0){
                continue;
            }
            VLOG(1) << "Received a gulp: "<<i;
            Payload pld;
            input["gulp"].pop(pld);

            // auto pld = input["gulp"].peek<Payload>();
            if (!pld) {
                VLOG(1) << "Null payload";
                return raft::proceed;
            }
            auto metadata = pld.get_mbuf()->get_metadata();
            int nchan = std::any_cast<uint8_t>((*metadata)["nchan"]);
            int chan0 = std::any_cast<int64_t>((*metadata)["chan0"]);
            VLOG(1) << "nchan: " << nchan << " chan0: " << chan0;
            // input["gulp"].recycle();
        }

        return raft::proceed;
    }
};