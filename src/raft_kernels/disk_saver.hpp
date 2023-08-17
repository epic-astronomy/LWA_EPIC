#ifndef DISK_SAVER
#define DISK_SAVER
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
namespace py = pybind11;
using namespace std::string_literals;

template<class Payload>
class DiskSaver_rft: public raft::kernel{

    public:
    DiskSaver_rft():raft::kernel(){
        input.addPort<Payload>("image");
    }

    virtual raft::kstatus run() override{
        Payload pld;
        input["image"].pop(pld);

        if(!pld){
            LOG(WARNING)<<"Empty image received.";
            return raft::proceed;
        }

        auto& img_metadata = pld.get_mbuf()->get_metadataref();
        for(auto it=img_metadata.begin();it!=img_metadata.end();++it){
            VLOG(3)<<it->first<<std::endl;
        }
        auto imsize = std::get<int>(img_metadata["grid_size"]);
        auto nchan = std::get<uint8_t>(img_metadata["nchan"]);
        save_image(imsize, nchan, pld.get_mbuf()->get_data_ptr(), "test_image"s, img_metadata);

        return raft::proceed;
    }
};


#endif /* DISK_SAVER */
