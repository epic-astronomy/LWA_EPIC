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
#include <glog/logging.h>
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

#include "raft_kernels/correlator.hpp"
#include "raft_kernels/disk_saver.hpp"
#include "raft_kernels/dummy_kernel.hpp"
#include "raft_kernels/packet_gen.hpp"
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
    py::scoped_interpreter guard{};
    py::gil_scoped_release release;
    FLAGS_logtostderr = 1;
    google::InitGoogleLogging(argv[0]);

    LOG(INFO) << "E-Field Parallel Imaging Correlator (EPIC) v" << EPIC_VERSION;
    
    std::string ip = "239.168.40.11";
    int port = 4015;

    auto gulper_ptr = std::make_unique<default_pkt_assembler>(ip, port);
    using payload_t = typename default_pkt_assembler::payload_t;

    auto correlator_options = MOFFCorrelatorDesc();
    auto corr_ptr = std::make_unique<MOFFCorrelator_t>(correlator_options);

    auto gulper_rft = GulpGen_rft<default_pkt_assembler>(gulper_ptr, 5);
    auto corr_rft = Correlator_rft<payload_t, MOFFCorrelator_t>(corr_ptr);
    auto saver_rft = DiskSaver_rft<payload_t>();
    dummy<default_pkt_assembler::payload_t> dummy_rft;

    raft::map m;

    m += gulper_rft >> corr_rft >> saver_rft;
    m.exe();

    LOG(INFO) << "END";

    return EXIT_SUCCESS;
}