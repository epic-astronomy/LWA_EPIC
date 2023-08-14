#ifndef EPIC_EXECUTOR
#define EPIC_EXECUTOR
#include "kernel_types.hpp"

template<unsigned int _GpuId>
class EPICKernels
{
  private:
    static constexpr unsigned int m_nkernels{ 8 };
    DummyPktGen_kt m_pkt_gen;
    EPICCorrelator_kt m_correlator;
    ChanReducer_kt m_chan_reducer;
    PixelExtractor_kt m_pixel_extractor;
    IndexFetcher_kt m_index_fetcher;
    DBIngester_kt m_db_ingester;
    Accumulator_kt m_accumulator;
    DiskSaver_kt m_disk_saver;

  protected:
    raft::map* m_map;
    void bind_kernels2cpu()
    {
        constexpr unsigned int cpu_ofst = (_GpuId) * m_nkernels;
        rft_manip<0 + cpu_ofst, 1>::bind(m_pkt_gen);
        rft_manip<1 + cpu_ofst, 1>::bind(m_correlator);
        rft_manip<2 + cpu_ofst, 1>::bind(m_chan_reducer);
        rft_manip<3 + cpu_ofst, 1>::bind(m_pixel_extractor);
        rft_manip<4 + cpu_ofst, 1>::bind(m_index_fetcher);
        rft_manip<5 + cpu_ofst, 1>::bind(m_db_ingester);
        rft_manip<6 + cpu_ofst, 1>::bind(m_accumulator);
        rft_manip<7 + cpu_ofst, 1>::bind(m_disk_saver);
    }

    void init_map()
    {
        auto& m = *m_map;
        m += m_pkt_gen >> m_correlator >> m_chan_reducer["in_img"]["out_img"] >> m_pixel_extractor["in_img"];

        m += m_pixel_extractor["out_img"] >> m_accumulator >> m_disk_saver;

        m += m_pixel_extractor["out_pix_rows"] >> m_db_ingester;

        m += m_chan_reducer["seq_start_id"] >> m_index_fetcher >> m_pixel_extractor["meta_pixel_rows"];
    }

  public:
    EPICKernels(KernelTypeDefs::opt_t& p_options, raft::map* p_map)
      : m_pkt_gen(get_dummy_pkt_gen_k<_GpuId>(p_options))
      , m_correlator(get_epiccorr_k<_GpuId>(p_options))
      , m_chan_reducer(get_chan_reducer_k<_GpuId>(p_options))
      , m_pixel_extractor(get_pixel_extractor_k<_GpuId>(p_options))
      , m_index_fetcher(get_index_fetcher_k<_GpuId>(p_options))
      , m_db_ingester(get_db_ingester_k<_GpuId>(p_options))
      , m_accumulator(get_accumulator_k<_GpuId>(p_options))
      , m_disk_saver(get_disk_saver_k<_GpuId>(p_options))
      , m_map(p_map)
    {
        LOG(INFO)<<"Binding CPUs";
        this->bind_kernels2cpu();
        LOG(INFO)<<"Initializing the map";
        this->init_map();
    }
};

template<unsigned int _nthGPU>
class EPIC : public EPICKernels<_nthGPU-1>
{
    //     using DummyPktGen_k = Kernel<DUMMY_PACK_GEN>::type;
    // using EPICCorrelator_k = Kernel<CORRELATOR>::type;
    // using ChanReducer_k = Kernel<CHAN_REDUCER>::type;
    // using PixelExtractor_k = Kernel<PIX_EXTRACTOR>::type;
    // using IndexFetcher_k = Kernel<IDX_FETCHER>::type;
    // using DBIngester_k = Kernel<DB_INGESTER>::type;
    // using Accumulator_k = Kernel<ACCUMULATOR>::type;
    // using DiskSaver_k = Kernel<DISK_SAVER>::type;

    //   private:
    //     DummyPktGen_kt m_pkt_gen;
    //     EPICCorrelator_kt m_correlator;
    //     ChanReducer_kt m_chan_reducer;
    //     PixelExtractor_kt m_pixel_extractor;
    //     IndexFetcher_kt m_index_fetcher;
    //     DBIngester_kt m_db_ingester;
    //     Accumulator_kt m_accumulator;
    //     DiskSaver_kt m_disk_saver;

  protected:
    raft::map* m_map;
    EPIC<_nthGPU - 1> m_next_epic;

  public:
    EPIC(KernelTypeDefs::opt_t& p_options, raft::map* p_map)
      : EPICKernels<_nthGPU-1>(p_options, p_map)
      , m_map(p_map)
      , m_next_epic(p_options, p_map)
    {
    }
};

template<>
class EPIC<1> : public EPICKernels<0>
{
  public:
    EPIC(KernelTypeDefs::opt_t& p_options, raft::map* p_map)
      : EPICKernels<0>(p_options, p_map)
    {
    }
};

void
run_epic(int argc, char** argv)
{
    auto option_list = get_options();
    auto options = option_list.parse(argc, argv);
    auto opt_valid = validate_options(options);

    LOG(INFO) << "Parsing options";
    if (opt_valid.value_or("none") != "none"s) {
        LOG(FATAL) << opt_valid.value();
    }

    int ngpus = options["ngpus"].as<int>();
    raft::map m;

    LOG(INFO)<<"Initializing EPIC";
    if (ngpus == 3) {
        auto epic = EPIC<3>(options, &m);
        m.exe();
    } else if (ngpus == 2) {
        auto epic = EPIC<2>(options, &m);
        m.exe();
    } else {
        auto epic = EPIC<1>(options, &m);
        m.exe();
    }
}

// template<EPICKernelID _kernel>
// typename Kernel<_kernel>::type
// get_kernel(cxxopts::ParseResult& options);

// template<>
// DummyPktGen_k
// get_kernel<DUMMY_PACK_GEN>(cxxopts::ParseResult& options)
// {
//     return DummyPktGen_k(2);
// }

// template<>
// EPICCorrelator_k
// get_kernel<CORRELATOR>(cxxopts::ParseResult& options)
// {
//     auto correlator_options = MOFFCorrelatorDesc();
//     // correlator_options.device_id = _GpuId;
//     correlator_options.accum_time_ms = options["seq_accum"].as<int>();
//     correlator_options.nseq_per_gulp = options["nts"].as<int>();
//     correlator_options.nchan_out = options["channels"].as<int>();
//     if (options["imagesize"].as<int>() == 64) {
//         correlator_options.img_size = HALF; // defaults to full
//     }
//     correlator_options.grid_res_deg = options["imageres"].as<float>();
//     correlator_options.support_size = options["support"].as<int>();
//     correlator_options.gcf_kernel_dim = std::sqrt(options["aeff"].as<float>()) * 10; // radius of the kernel in decimeters
//     correlator_options.kernel_oversampling_factor = options["kernel_oversample"].as<int>();
//     correlator_options.use_bf16_accum = options["accum_16bit"].as<bool>();
//     correlator_options.nstreams = options["nstreams"].as<int>();

//     auto corr_ptr = std::make_unique<MOFFCorrelator_t>(correlator_options);
//     return EPICCorrelator_k(corr_ptr);
// }

// template<>
// ChanReducer_k
// get_kernel<CHAN_REDUCER>(cxxopts::ParseResult& options)
// {
//     int imsize = options["imagesize"].as<int>();
//     int im_nchan = options["channels"].as<int>();
//     int chan_nbin = options["chan_nbin"].as<int>();
//     int reduced_nchan = im_nchan / chan_nbin;
//     int im_naccum = options["nimg_accum"].as<int>();
//     return ChanReducer_k(
//       chan_nbin, imsize, imsize, im_nchan);
// }

// template<>
// PixelExtractor_k
// get_kernel<PIX_EXTRACTOR>(cxxopts::ParseResult& options)
// {
//     KernelTypeDefs::pixel_buf_config_t config;
//     int imsize = options["imagesize"].as<int>();
//     int im_nchan = options["channels"].as<int>();
//     int chan_nbin = options["chan_nbin"].as<int>();
//     int reduced_nchan = im_nchan / chan_nbin;
//     config.nchan = reduced_nchan;
//     config.ncoords = 1;
//     config.nsrcs = 1;

//     // fetch intial pixel indices;
//     auto dummy_meta = create_dummy_meta(imsize, imsize);
//     return PixelExtractor_k(config, dummy_meta, imsize, imsize, reduced_nchan);
// }

// template<>
// IndexFetcher_k
// get_kernel<IDX_FETCHER>(cxxopts::ParseResult& options)
// {
//     return IndexFetcher_k();
// }

// template<>
// DBIngester_k
// get_kernel<DB_INGESTER>(cxxopts::ParseResult& options)
// {
//     return DBIngester_k();
// }

// template<>
// Accumulator_k
// get_kernel<ACCUMULATOR>(cxxopts::ParseResult& options)
// {
//     int imsize = options["imagesize"].as<int>();
//     int im_nchan = options["channels"].as<int>();
//     int chan_nbin = options["chan_nbin"].as<int>();
//     int reduced_nchan = im_nchan / chan_nbin;
//     int im_naccum = options["nimg_accum"].as<int>();

//     auto accumulator_rft = Accumulator_k(
//       imsize, imsize, reduced_nchan, im_naccum);
// }

// template<>
// DiskSaver_k
// get_kernel<DISK_SAVER>(cxxopts::ParseResult& options)
// {
//     return DiskSaver_k();
// }
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

#endif /* EPIC_EXECUTOR */
