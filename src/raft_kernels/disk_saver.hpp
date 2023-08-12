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
    protected:
    // uint64_t m_time_from_unix_epoch_s{0};
    // uint64_t time_tag0{0};
    public:
    DiskSaver_rft(/*std::string utcstart="2023_06_19T00_00_00"s*/):raft::kernel(){
        input.addPort<Payload>("image");
        // if(utcstart==""){
        //     m_time_from_unix_epoch_s = get_ADP_time_from_unix_epoch();
        // }
        // else{
        //     m_time_from_unix_epoch_s = get_time_from_unix_epoch(utcstart);
        // }
        // time_tag0 = m_time_from_unix_epoch_s * FS;

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
        for(auto it=img_metadata.begin();it!=img_metadata.end();++it){
            std::cout<<it->first<<std::endl;
        }
        auto imsize = std::get<int>(img_metadata["grid_size"]);
        auto nchan = std::get<uint8_t>(img_metadata["nchan"]);
        // img_metadata["time_tag"] = time_tag0 + std::get<uint64_t>(img_metadata["seq_start"]) * SEQ_MULT_S;
        save_image(imsize, nchan, pld.get_mbuf()->get_data_ptr(), "test_image"s, img_metadata);

        return raft::proceed;
    }
};