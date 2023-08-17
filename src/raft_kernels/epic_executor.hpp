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
        constexpr unsigned int cpu_ofst = (_GpuId) * m_nkernels + 2 /*for the runtime*/;
        // ensure the cpu ID is always non-zero.
        // Setting it to zero causes instability
        rft_manip<1 + cpu_ofst, 1>::bind(m_pkt_gen);
        rft_manip<2 + cpu_ofst, 1>::bind(m_correlator);
        rft_manip<3 + cpu_ofst, 1>::bind(m_chan_reducer);
        rft_manip<4 + cpu_ofst, 1>::bind(m_pixel_extractor);
        rft_manip<5 + cpu_ofst, 1>::bind(m_index_fetcher);
        rft_manip<6 + cpu_ofst, 1>::bind(m_db_ingester);
        rft_manip<7 + cpu_ofst, 1>::bind(m_accumulator);
        rft_manip<8 + cpu_ofst, 1>::bind(m_disk_saver);
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
