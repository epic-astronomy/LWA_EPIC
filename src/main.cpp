
#include <cmath>
#include "ex/MOFF_correlator.hpp"
#include "ex/option_parser.hpp"

#include "ex/buffer.hpp"
#include "raft_kernels/correlator.hpp"
#include "raft_kernels/disk_saver.hpp"
#include "raft_kernels/dummy_kernel.hpp"
#include "raft_kernels/dummy_packet_gen.hpp"
#include "ex/lf_buf_mngr.hpp"

using namespace std::chrono;
using namespace std::string_literals;
namespace hn = hwy::HWY_NAMESPACE;
using tag8 = hn::ScalableTag<uint8_t>;

// int
// get_chan0(std::string ip, int port)
// {

//     // std::cout<<"receiving\n";
//     auto receiver = VMAReceiver<uint8_t, AlignedBuffer, MultiCastUDPSocket, REG_COPY>();
//     uint8_t* buf;
//     receiver.set_address(ip, port);
//     receiver.bind_socket();
//     int nbytes = receiver.recv_packet(buf);
//     // std::cout<<nbytes;
//     const chips_hdr_type* pkt_hdr = (chips_hdr_type*)buf;
//     return (ntohs(pkt_hdr->chan0));
// }

namespace py = pybind11;
// #define _VMA_ 1
int
main(int argc, char** argv)
{
    py::scoped_interpreter guard{};
    py::gil_scoped_release release;
    FLAGS_logtostderr = 1;
    google::InitGoogleLogging(argv[0]);

    auto option_list = get_options();
    VLOG(3)<<"Parsing options";
    auto options = option_list.parse(argc,argv);
    if(options.count("help")){
        std::cout<<option_list.help()<<std::endl;
        exit(0);
    }

    VLOG(3)<<"Validating options";
    auto opt_valid = validate_options(options);

    if(opt_valid.value_or("none")!="none"s){
        std::cout<<opt_valid.value()<<"\n";
        exit(0);
    }

    LOG(INFO) << "E-Field Parallel Imaging Correlator (EPIC) v" << EPIC_VERSION;
    
    // std::string ip = "239.168.40.11";
    // int port = 4015;

    // auto gulper_ptr = std::make_unique<default_pkt_assembler>(ip, port);
    // LFBufMngr<AlignedBuffer<uint8_t>>

    auto correlator_options = MOFFCorrelatorDesc();
    correlator_options.accum_time_ms = options["accumulate"].as<int>();
    correlator_options.nseq_per_gulp = options["nts"].as<int>();
    correlator_options.nchan_out = options["channels"].as<int>();
    if(options["imagesize"].as<int>()==64){
        correlator_options.img_size=HALF;//defaults to full
    }
    correlator_options.grid_res_deg=options["imageres"].as<float>();
    correlator_options.support_size=options["support"].as<int>();
    correlator_options.gcf_kernel_dim=std::sqrt(options["aeff"].as<float>())*10;//radius of the kernel in decimeters
    correlator_options.kernel_oversampling_factor = options["kernel_oversample"].as<int>();


    auto corr_ptr = std::make_unique<MOFFCorrelator_t>(correlator_options);

    using mbuf_t = typename LFBufMngr<AlignedBuffer<uint8_t>>::mbuf_t;
    using payload_t = Payload<mbuf_t>;
    // auto gulper_rft = GulpGen_rft<default_pkt_assembler>(gulper_ptr, 5);
    VLOG(1)<<"Initializing the Raft kernels";
    VLOG(1)<<"Correlator";
    auto corr_rft = Correlator_rft<payload_t, MOFFCorrelator_t>(corr_ptr);
    VLOG(1)<<"Saver";
    auto saver_rft = DiskSaver_rft<MOFFCorrelator_t::payload_t>();
    // dummy<default_pkt_assembler::payload_t> dummy_rft;
    VLOG(1)<<"Dummy packet generator";
    auto dummy_pkt_gen_rft =  dummy_pkt_gen<payload_t,LFBufMngr<AlignedBuffer<uint8_t>>>(2);

    VLOG(1)<<"Setting up the Raft map";
    raft::map m;

    m += dummy_pkt_gen_rft>> corr_rft >> saver_rft;
    VLOG(1)<<"Done";
    m.exe();

    LOG(INFO) << "END";

    return EXIT_SUCCESS;
}