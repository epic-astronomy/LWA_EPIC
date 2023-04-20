#include "../ex/constants.h"
#include "../ex/types.hpp"
#include <chrono>
#include <glog/logging.h>
#include "../ex/py_funcs.hpp"
#include <memory>
#include <raft>
#include <cstring>
#include <raftio>
#include <string>

using namespace std::string_literals;

template<class Payload>
class DiskSaver_rft: public raft::kernel{
    protected:
    double m_ADP_time_from_unix_epoch_s{0};
    public:
    DiskSaver_rft():raft::kernel(){
        input.addPort<Payload>("image");
        //m_ADP_time_from_unix_epoch_s = get_ADP_time_from_unix_epoch();
    }

    virtual raft::kstatus run() override{
        VLOG(2)<<"Inside saver rft";

        Payload pld;
        input["image"].pop(pld);

        if(!pld){
            LOG(WARNING)<<"Empty image received.";
            return raft::proceed;
        }

        auto& img_metadata = pld.get_mbuf()->get_metadataref();
        auto imsize = std::any_cast<int>(img_metadata["grid_size"]);
        auto nchan = std::any_cast<uint8_t>(img_metadata["nchan"]);
        save_image(imsize, nchan, pld.get_mbuf()->get_data_ptr(), "test_image.png"s);

        return raft::proceed;
    }
};