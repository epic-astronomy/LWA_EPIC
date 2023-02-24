#include "ex/bf_ibverbs.hpp"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/highway.h"
#include "hwy/print.h"
// #include "infinity/infinity.h"
#include "ex/MOFF_correlator.hpp"
#include "ex/buffer.hpp"
#include "ex/exceptions.hpp"
#include "ex/lf_buf_mngr.hpp"
#include "ex/packet_assembler.hpp"
#include "ex/packet_receiver.hpp"
#include "ex/py_funcs.hpp"
#include "ex/sockets.h"
#include <arpa/inet.h>
#include <bits/types/struct_iovec.h>
#include <bitset>
#include <cmath>
#include <cstdlib>
#include <infiniband/verbs.h>
#include <mellanox/vma_extra.h>
#include <netdb.h>
#include <netinet/in.h>
#include <pybind11/embed.h>
#include <raft>
#include <raftio>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <x86intrin.h>
#include <glog/logging.h>


#include "raft_kernels/packet_gen.hpp"
#include "raft_kernels/dummy_kernel.hpp"
// #include "ex/packet_assembler.h"
using namespace std::chrono;
using namespace std::string_literals;
namespace hn = hwy::HWY_NAMESPACE;
using tag8 = hn::ScalableTag<uint8_t>;


int
get_chan0(std::string ip, int port)
{

    // std::cout<<"receiving\n";
    auto receiver = VMAReceiver<uint8_t, AlignedBuffer, MultiCastUDPSocket, REG_COPY>();
    uint8_t* buf;
    receiver.set_address(ip, port);
    receiver.bind_socket();
    int nbytes = receiver.recv_packet(buf);
    // std::cout<<nbytes;
    const chips_hdr_type* pkt_hdr = (chips_hdr_type*)buf;
    return (ntohs(pkt_hdr->chan0));
}

namespace py = pybind11;

// #define _VMA_ 1
int
main(int argc, char** argv)
{   
    FLAGS_logtostderr = 1;
    google::InitGoogleLogging(argv[0]);


    LOG(INFO)<<"E-Field Parallel Imaging Correlator (EPIC) v"<<EPIC_VERSION;
    if(VLOG_IS_ON(1)){
        VLOG(1)<<"Verbose logging";
    }
    else{
        LOG(INFO)<<"VLOG(1) is off";
    }
    // std::cout<<pro_sph_cv(0,0,8.413185126313465);
    // return 0;
    std::string ip = "239.168.40.11";
    int port = 4015;

    LOG(INFO)<<"Initializing Verbs packet assembler";
    // auto gulper = verbs_pkt_assembler(ip, port);
    DLOG(INFO) <<"Number of 8-bit lanes"<<int(HWY_LANES(uint8_t));
    LOG(INFO)<<"Initializing MOFF Correlator";
    auto gulper_ptr = std::make_unique<default_pkt_assembler>(ip, port);
    //const auto gulp = gulper_ptr.get()->get_gulp();
    // gulp.mbuf_shared_count();
    //DLOG(INFO)<<"Got the gulp";

    auto gulper_rft = GulpGen_rft<default_pkt_assembler>(gulper_ptr, 5);
    dummy<default_pkt_assembler::payload_t> dummy_rft;

    raft::map m;

    m+= gulper_rft >> dummy_rft;
    m.exe();

    LOG(INFO)<<"END";

    return 0;

#ifdef _UNUSED_

    MOFFCorrelator_t moff_correlator(1.5, 1000, 2, 64, 2.0, 8);
    // py::scoped_interpreter guard{};

    // std::cout << "wtf\n";
    // // return 0;
    // std::cout << "test\n";
    // try {
    //     auto lf_storage = LFBufMngr<AlignedBuffer<uint8_t>>(2, 2);
    //     std::cout << "1\n";
    //     auto buf1 = lf_storage.acquire_buf();
    //     auto buf2 = lf_storage.acquire_buf();
    //     auto buf3 = lf_storage.acquire_buf();
    // } catch (const InvalidSize& e) {
    //     e.print();
    // }
    // // auto mbuf = buf1.get_mbuf();
    // //  auto buf4 = buf1;
    // //  return 0;


    // std::cout << "Fine\n";
    // auto gulper = default_pkt_assembler(ip, port);
    bool show_freqs = false;

    py::scoped_interpreter guard{};

    auto start = high_resolution_clock::now();

    auto stop = high_resolution_clock::now();
    auto begin = high_resolution_clock::now();
    // std::cout << "assembler start\n";
    auto scipy_spl = py::module_::import("scipy.special");
    start = high_resolution_clock::now();
    auto cv = pro_sph_cv(scipy_spl, 0, 0, 8.413185126313465);
    stop = high_resolution_clock::now();
    std::cout << "cv time: " << duration_cast<microseconds>(stop - start).count() << "\n";
    std::cout << cv << "\n";
    std::cout << pro_sph_ang1_cv(scipy_spl, 0, 0, 8.413185126313465, cv, 0.5) << "\n";
    // py::finalize_interpreter();
    auto stands = hwy::AllocateAligned<double>(LWA_SV_NSTANDS * 3);
    // auto delta = get_lwasv_locs<double>(stands.get(), 64, 1.8975);
    // std::cout<<"delta: "<<delta<<"\n";
    // std::cout<<stands[0]<<" "<<stands[1]<<" "<<stands[2]<<"\n";

    // return 0;

    for (auto i = 0; i < 50; ++i) {
        std::cout << "i: " << i << "\n";
        if (i == 40)
            begin = high_resolution_clock::now();
        try {
            start = high_resolution_clock::now();
            std::cout << "gap: " << duration_cast<milliseconds>(start - stop).count() << "\n";
            auto gulp = gulper.get_gulp();
            stop = high_resolution_clock::now();
            std::cout << "40ms gulp time: " << duration_cast<milliseconds>(stop - start).count() << " " << i << "\n";

            if (i ==49 && gulp.get_mbuf() != nullptr) {
                auto metadata = gulp.get_mbuf()->get_metadata();
                if (metadata == nullptr) {
                    std::cout << "empty metadata\n";
                }
                for (auto& it : *metadata) {
                    std::cout << it.first << "\n";
                }
                int nchan = std::any_cast<uint8_t>((*metadata)["nchan"]);
                int chan0 = std::any_cast<int64_t>((*metadata)["chan0"]);
                std::cout << "nchan: " << nchan << " chan0: " << chan0 << "\n";
                //moff_correlator.reset(nchan, chan0, 2, 64, 2.0, 32, 1000);
                moff_correlator.allocate_f_eng_gpu(gulp.get_mbuf()->buf_size());
                int npixels = 64*64*nchan;
                moff_correlator.allocate_out_img(npixels*sizeof(float));
                auto output = hwy::AllocateAligned<float>(npixels);
                auto dptr = gulp.get_mbuf()->get_data_ptr();
                moff_correlator.process_gulp(dptr, gulp.get_mbuf()->buf_size(), output.get(), npixels*sizeof(float));
                save_image(64, nchan, output.get(), "test_image.png"s);
                // for (int j = 0; j < 10; ++j)
                //     std::cout << "Feng data cpu: " << int(dptr[j]) << " " << int(dptr[j + 1]) << "\n"; 
            }
        } catch (const PacketReceiveFailure& err) {
            err.print();
            std::cout << "failed\n";
            // err.print();
        }
    }
    auto end = high_resolution_clock::now();
    std::cout << "Total time (ms): " << duration_cast<milliseconds>(end - begin).count() << "\n";

    return 0;

    auto receiver = VMAReceiver<uint8_t, AlignedBuffer, MultiCastUDPSocket>();

    auto ip_addrs = std::vector<std::string>{
        "239.168.40.11",
        "239.168.40.12",
        "239.168.40.13",
        "239.168.40.14",
        "239.168.40.15",
        "239.168.40.16"
    };

    auto ports = std::vector<int>{ 4015, 4016 };

    if (show_freqs) {
        for (auto ip : ip_addrs) {
            for (auto port : ports) {
                auto chan0 = get_chan0(ip, port);
                std::cout << ip << ":" << port << ": " << float(chan0) * 0.025 << " MHz\n";
            }
        }
        return (0);
    }
    
    return (EXIT_SUCCESS);
    // #endif
#endif
}