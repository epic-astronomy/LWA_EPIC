
#include "ex/MOFF_correlator.hpp"
#include "ex/option_parser.hpp"
#include <cmath>

#include "ex/buffer.hpp"
#include "ex/helper_traits.hpp"
#include "ex/lf_buf_mngr.hpp"
#include "raft_kernels/chan_reducer.cpp"
#include "raft_kernels/correlator.hpp"
#include "raft_kernels/disk_saver.hpp"
#include "raft_kernels/dummy_kernel.hpp"
#include "raft_kernels/dummy_packet_gen.hpp"
#include "raft_kernels/accumulator.cpp"
#include "raft_kernels/pixel_extractor.cpp"
#include "raft_kernels/db_ingester.cpp"
#include "raft_kernels/index_fetcher.hpp"
#include <raftmanip>

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
    google::EnableLogCleaner(3);

    auto option_list = get_options();
    VLOG(3) << "Parsing options";
    auto options = option_list.parse(argc, argv);
    if (options.count("help")) {
        std::cout << option_list.help() << std::endl;
        exit(0);
    }

    VLOG(3) << "Validating options";
    auto opt_valid = validate_options(options);

    if (opt_valid.value_or("none") != "none"s) {
        std::cout << opt_valid.value() << "\n";
        exit(0);
    }

    LOG(INFO) << "E-Field Parallel Imaging Correlator (EPIC) v" << EPIC_VERSION;

    // std::string ip = "239.168.40.11";
    // int port = 4015;

    // auto gulper_ptr = std::make_unique<default_pkt_assembler>(ip, port);
    // LFBufMngr<AlignedBuffer<uint8_t>>

    auto correlator_options = MOFFCorrelatorDesc();
    correlator_options.accum_time_ms = options["seq_accum"].as<int>();
    correlator_options.nseq_per_gulp = options["nts"].as<int>();
    correlator_options.nchan_out = options["channels"].as<int>();
    if (options["imagesize"].as<int>() == 64) {
        correlator_options.img_size = HALF; // defaults to full
    }
    correlator_options.grid_res_deg = options["imageres"].as<float>();
    correlator_options.support_size = options["support"].as<int>();
    correlator_options.gcf_kernel_dim = std::sqrt(options["aeff"].as<float>()) * 10; // radius of the kernel in decimeters
    correlator_options.kernel_oversampling_factor = options["kernel_oversample"].as<int>();
    correlator_options.use_bf16_accum = options["accum_16bit"].as<bool>();
    correlator_options.nstreams = options["nstreams"].as<int>();

    auto corr_ptr = std::make_unique<MOFFCorrelator_t>(correlator_options);

    using mbuf_t = typename LFBufMngr<AlignedBuffer<uint8_t>>::mbuf_t;
    using payload_t = Payload<mbuf_t>;
    using float_buf_t = LFBufMngr<AlignedBuffer<float>>;
    // auto gulper_rft = GulpGen_rft<default_pkt_assembler>(gulper_ptr, 5);
    VLOG(1) << "Initializing the Raft kernels";
    VLOG(1) << "Correlator";
    auto corr_rft = Correlator_rft<payload_t, MOFFCorrelator_t>(corr_ptr);
    VLOG(1) << "Saver";
    auto saver_rft = DiskSaver_rft<MOFFCorrelator_t::payload_t>();
    // dummy<default_pkt_assembler::payload_t> dummy_rft;
    VLOG(1) << "Dummy packet generator";
    auto dummy_pkt_gen_rft = dummy_pkt_gen<payload_t, LFBufMngr<AlignedBuffer<uint8_t>>>(2);

    int imsize = options["imagesize"].as<int>();
    int im_nchan = options["channels"].as<int>();
    int chan_nbin = options["chan_nbin"].as<int>();
    int reduced_nchan = im_nchan/chan_nbin;
    int im_naccum = options["nimg_accum"].as<int>();
    auto chan_reducer_rft = ChanReducer_rft<MOFFCorrelator_t::payload_t, float_buf_t>(
     chan_nbin , imsize, imsize, im_nchan);

    auto accumulator_rft = Accumulator_rft<MOFFCorrelator_t::payload_t>(
        imsize, imsize, reduced_nchan, im_naccum
    );

    using pixel_buf_t = LFBufMngr<EpicPixelTableDataRows<float>>;
    using pixel_buf_config_t = typename EpicPixelTableDataRows<float>::config_t;
    using pix_pld_t = Payload<typename pixel_buf_t::mbuf_t>;
    pixel_buf_config_t config;
    config.nchan=reduced_nchan;
    config.ncoords=1;
    auto dummy_meta = create_dummy_meta(imsize, imsize);
    auto pixel_extractor_rft = PixelExtractor<MOFFCorrelator_t::payload_t, pix_pld_t, pixel_buf_t, pixel_buf_config_t>(config, dummy_meta, imsize, imsize, reduced_nchan);

    auto index_fetcher_rft = IndexFetcher_rft();

    auto db_injester_rft = DBIngester_rft<pix_pld_t>();

    VLOG(1) << "Setting up the Raft map";
    raft::map m;

    // CPU id, Affinity group id
    rft_manip<1, 1>::bind(dummy_pkt_gen_rft);
    rft_manip<2, 1>::bind(corr_rft);
    rft_manip<3, 1>::bind(saver_rft);
    rft_manip<4, 1>::bind(chan_reducer_rft);
    rft_manip<5, 1>::bind(accumulator_rft);
    rft_manip<6, 1>::bind(pixel_extractor_rft);
    rft_manip<7, 1>::bind(db_injester_rft);
    rft_manip<8, 1>::bind(index_fetcher_rft);



    m += dummy_pkt_gen_rft >> corr_rft >> chan_reducer_rft["in_img"]["out_img"] >> pixel_extractor_rft["in_img"];

    m+= pixel_extractor_rft["out_img"]>> accumulator_rft >> saver_rft;

    m+= pixel_extractor_rft["out_pix_rows"] >> db_injester_rft;

    m+= chan_reducer_rft["seq_start_id"] >> index_fetcher_rft >> pixel_extractor_rft["meta_pixel_rows"];

    VLOG(1) << "Done";
    m.exe();

    LOG(INFO) << "END";

    return EXIT_SUCCESS;
}
