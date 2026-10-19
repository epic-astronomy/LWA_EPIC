/*
 Copyright (c) 2023 The EPIC++ authors

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

#ifndef SRC_RAFT_KERNELS_EPIC_EXECUTOR_HPP_
#define SRC_RAFT_KERNELS_EPIC_EXECUTOR_HPP_
#include <memory>
#include <string>
#include <utility>

#include "../ex/cpu_reserve.hpp"
#include "../ex/metrics.hpp"
#include "../ex/station_desc.hpp"
#include "./bench_kernels.hpp"
#include "./vistap_kernels.hpp"
#include "./kernel_types.hpp"
#include "../ex/option_parser.hpp"

// #include "./epic_live_streamer.hpp"

/// Benchmark settings from the --bench_* options
inline BenchOptions GetBenchOptions(const KernelTypeDefs::opt_t& p_options) {
  BenchOptions bo;
  bo.data_path = p_options["bench_data"].as<std::string>();
  bo.chan0 = p_options["bench_chan0"].as<int64_t>();
  bo.nchan = NDP_NCHAN_SELECT;
  bo.nseq = p_options["nts"].as<int>();
  bo.seq0 = BenchFirstSeq(p_options);
  bo.ngulps = p_options["bench_ngulps"].as<int64_t>();
  bo.realtime = p_options["bench_realtime"].as<bool>();
  bo.rotate = p_options["bench_rotate"].as<int>();
  bo.dump_dir = p_options["bench_dump"].as<std::string>();
  bo.dump_every = p_options["bench_dump_every"].as<int>();
  bo.report_every = p_options["bench_report"].as<int>();
  bo.drop = p_options["bench_drop"].as<double>();
  return bo;
}
using BenchGen_t = BenchGulpGen<KernelTypeDefs::payload_u8_t, LFBufMngr<AlignedBuffer<uint8_t>>>;

template <unsigned int _GpuId>
class EPICKernels {
 private:
  std::unique_ptr<Streamer> m_streamer;
  bool m_is_offline{false};
  /// --bench --bench_full: generated gulps in place of the packet capture
  std::unique_ptr<BenchGen_t> m_bench_gen;
  static constexpr unsigned int m_nkernels{9};
  PktGen_kt m_pkt_gen;
  DummyPktGen_kt m_dpkt_gen;
  EPICCorrelator_kt m_correlator;
  ChanReducer_kt m_chan_reducer;
  PixelExtractor_kt m_pixel_extractor;
  IndexFetcher_kt m_index_fetcher;
  DBIngester_kt m_db_ingester;
  Accumulator_kt m_accumulator;
  DiskSaver_kt m_disk_saver;
  EpicLiveStream_kt m_live_streamer;

 protected:
  raft::map* m_map;
  void BindKernels2Cpu(const KernelTypeDefs::opt_t& p_options) {
    if (p_options["cpu_layout"].as<std::string>() == "compact") {
      // 3-7 for GPU 0: the busy kernels alone, the light ones together
      constexpr unsigned int b = (_GpuId)*5 + 3;
      if (m_bench_gen) {
        RftManip<b, 1 + _GpuId>::bind(*m_bench_gen);
      } else if (m_is_offline) {
        RftManip<b, 1 + _GpuId>::bind(m_dpkt_gen);
      } else {
        RftManip<b, 1 + _GpuId>::bind(*(m_pkt_gen.get()));
      }
      RftManip<b + 1, 1 + _GpuId>::bind(m_correlator);
      RftManip<b + 2, 1 + _GpuId>::bind(m_chan_reducer);
      RftManip<b + 3, 1 + _GpuId>::bind(m_live_streamer);
      RftManip<b + 4, 1 + _GpuId>::bind(m_pixel_extractor);
      RftManip<b + 4, 1 + _GpuId>::bind(m_index_fetcher);
      RftManip<b + 4, 1 + _GpuId>::bind(m_db_ingester);
      RftManip<b + 4, 1 + _GpuId>::bind(m_accumulator);
      RftManip<b + 4, 1 + _GpuId>::bind(m_disk_saver);
      ReserveReceiver(b);
      return;
    }
    // use an additional offset of 2 for the runtime
    constexpr unsigned int cpu_ofst = (_GpuId)*m_nkernels + 2;
    // ensure the cpu ID is always non-zero.
    // Setting it to zero causes instability
    if (m_bench_gen) {
      RftManip<1 + cpu_ofst, 1 + _GpuId>::bind(*m_bench_gen);
    } else if (m_is_offline) {
      RftManip<1 + cpu_ofst, 1 + _GpuId>::bind(m_dpkt_gen);
    } else {
      RftManip<1 + cpu_ofst, 1 + _GpuId>::bind(*(m_pkt_gen.get()));
    }
    RftManip<2 + cpu_ofst, 1 + _GpuId>::bind(m_correlator);
    RftManip<3 + cpu_ofst, 1 + _GpuId>::bind(m_chan_reducer);
    RftManip<4 + cpu_ofst, 1 + _GpuId>::bind(m_pixel_extractor);
    RftManip<5 + cpu_ofst, 1 + _GpuId>::bind(m_index_fetcher);
    RftManip<6 + cpu_ofst, 1 + _GpuId>::bind(m_db_ingester);
    RftManip<7 + cpu_ofst, 1 + _GpuId>::bind(m_accumulator);
    RftManip<8 + cpu_ofst, 1 + _GpuId>::bind(m_disk_saver);
    RftManip<9 + cpu_ofst, 1 + _GpuId>::bind(m_live_streamer);
    ReserveReceiver(1 + cpu_ofst);
  }

  /// Keep the helper threads off the packet receiver's CPU and its sibling
  /// (cpu_reserve.hpp); threads started from here on take the rest
  void ReserveReceiver(int p_cpu) {
    if (m_bench_gen || m_is_offline) return;
    cpus::Reserve(p_cpu);
    cpus::UseFree();
    LOG(INFO) << "CPU " << p_cpu << " and its siblings reserved for the packet receiver";
  }

  void InitMap(const KernelTypeDefs::opt_t& p_options) {
    auto& m = *m_map;
    // m += m_dpkt_gen >> m_correlator >> m_disk_saver;
    if (m_bench_gen) {
      m += *m_bench_gen >> m_correlator["gulp"]["img"] >>
           m_chan_reducer["in_img"]["out_img"] >> m_pixel_extractor["in_img"];
    } else if (m_is_offline) {
      m += m_dpkt_gen >> m_correlator["gulp"]["img"] >>
           m_chan_reducer["in_img"]["out_img"] >> m_pixel_extractor["in_img"];
    } else {
      m += *(m_pkt_gen.get()) >> m_correlator["gulp"]["img"] >>
           m_chan_reducer["in_img"]["out_img"] >> m_pixel_extractor["in_img"];
    }
    m += m_correlator["img_stream"] >> m_live_streamer["in_img"];

    m += m_pixel_extractor["out_img"] >> m_accumulator >> m_disk_saver;

    m += m_pixel_extractor["out_pix_rows"] >> m_db_ingester;

    m += m_chan_reducer["seq_start_id"] >> m_index_fetcher >>
         m_pixel_extractor["meta_pixel_rows"];
    LOG(INFO) << "Done initializing the map";
  }

 public:
  EPICKernels(const KernelTypeDefs::opt_t& p_options, raft::map* p_map)
      : m_streamer(GetStreamer(p_options)),
        m_dpkt_gen(get_dummy_pkt_gen_k<_GpuId>(p_options)),
        m_correlator(get_epiccorr_k<_GpuId>(p_options)),
        m_chan_reducer(get_chan_reducer_k<_GpuId>(p_options)),
        m_pixel_extractor(get_pixel_extractor_k<_GpuId>(p_options)),
        m_index_fetcher(get_index_fetcher_k<_GpuId>(p_options)),
        m_db_ingester(get_db_ingester_k<_GpuId>(p_options)),
        m_accumulator(get_accumulator_k<_GpuId>(p_options)),
        m_disk_saver(get_disk_saver_k<_GpuId>(p_options)),
        m_live_streamer(get_epic_live_stream_k<_GpuId>(p_options)),
        m_map(p_map) {
    m_is_offline = p_options["offline"].as<bool>();
    // auto streamer = std::make_unique<Streamer>();
    LOG(INFO) << "Before pkt_gen";
    if (p_options["bench"].as<bool>() && p_options["bench_full"].as<bool>()) {
      auto bo = GetBenchOptions(p_options);
      bo.exit_grace_s = 10;
      m_bench_gen = std::make_unique<BenchGen_t>(bo);
      LOG(INFO) << "bench: the full pipeline on generated gulps, chan0 " << bo.chan0 << ", seq0 " << bo.seq0 << ", "
                << bo.ngulps << " gulps" << (bo.realtime ? ", real-time pacing" : ", unpaced");
    } else if (!m_is_offline) {
      m_pkt_gen = std::move(get_pkt_gen_k<_GpuId>(p_options));
    }
    // m_disk_saver.SetStreamer(m_streamer.get());
    m_live_streamer.SetStreamer(m_streamer.get());
    if(p_options.count("include_chans")){
      m_correlator.SetInclChans(p_options["include_chans"].as<std::vector<int>>());
    }

    LOG_IF(INFO, m_is_offline) << "EPIC will run on offline data";
    LOG(INFO) << "Binding kernels to CPUs";
    BindKernels2Cpu(p_options);
    LOG(INFO) << "Initializing the EPIC graph to run on GPU:" << _GpuId;
    InitMap(p_options);
  }
};

template <unsigned int _nthGPU>
class EPIC : public EPICKernels<_nthGPU - 1> {
 protected:
  raft::map* m_map;
  EPIC<_nthGPU - 1> m_next_epic;

 public:
  EPIC(const KernelTypeDefs::opt_t& p_options, raft::map* p_map)
      : EPICKernels<_nthGPU - 1>(p_options, p_map),
        m_map(p_map),
        m_next_epic(p_options, p_map) {}
};

template <>
class EPIC<1> : public EPICKernels<0> {
 public:
  EPIC(const KernelTypeDefs::opt_t& p_options, raft::map* p_map)
      : EPICKernels<0>(p_options, p_map) {}
};

/// `--bench`: generator -> CorrelatorRft (tap, gain table) -> sinks, on GPU 0.
/// Only these kernels are constructed, so nothing connects to the database,
/// the video server or the network.
void RunEpicBench(const KernelTypeDefs::opt_t& p_options) {
  BenchOptions bo = GetBenchOptions(p_options);
  bo.exit_grace_s = 10;  // the FX imager holds the last image (see bench_kernels.hpp)
  using gen_t = BenchGen_t;
  using sink_t = BenchSink<KernelTypeDefs::moffcorr_t::payload_t>;
  raft::map m;
  gen_t gen(bo);
  auto corr = get_epiccorr_k<0>(p_options);
  sink_t sink_img("img", bo, true);
  sink_t sink_stream("img_stream", bo, false);
  RftManip<3, 1>::bind(gen);
  RftManip<4, 1>::bind(corr);
  RftManip<5, 1>::bind(sink_img);
  RftManip<6, 1>::bind(sink_stream);
  // The production ChanReducer (with --sky_shm, the sky ring) between the correlator and the sink
  auto reducer = get_chan_reducer_k<0>(p_options);
  BenchSeqSink seq_sink;
  RftManip<7, 1>::bind(reducer);
  RftManip<8, 1>::bind(seq_sink);
  m += gen >> corr["gulp"]["img"] >> reducer["in_img"]["out_img"] >> sink_img;
  m += reducer["seq_start_id"] >> seq_sink;
  m += corr["img_stream"] >> sink_stream;
  LOG(INFO) << "bench: chan0 " << bo.chan0 << ", seq0 " << bo.seq0 << ", " << bo.ngulps << " gulps"
            << (bo.realtime ? ", real-time pacing" : ", unpaced");
  m.exe();
}

// Receiver + GPU calibration tap only (see vistap_kernels.hpp)
void RunEpicVisTap(const KernelTypeDefs::opt_t& p_options) {
  int dev = 0;
  if (p_options["gpu_ids"].count() > 0 && !p_options["gpu_ids"].as<std::vector<int>>().empty()) {
    dev = p_options["gpu_ids"].as<std::vector<int>>()[0];
  }
  auto with_gpu = [&](std::string p) {
    auto pos = p.find("{gpu}");
    if (pos != std::string::npos) p.replace(pos, 5, std::to_string(dev));
    return p;
  };
  TccTapOptions tap;
  tap.path = with_gpu(p_options["caltap_out"].as<std::string>());
  tap.interval_s = p_options["caltap_interval"].as<float>();
  tap.min_quality = p_options["caltap_min_quality"].as<float>();
  tap.record_dir = p_options["caltap_record_dir"].as<std::string>();
  const int nchan = p_options["channels"].as<int>();
  const int nseq = p_options["nts"].as<int>();
  LOG(INFO) << "vistap: receiver + tensor-core tap only, GPU " << dev << ", " << nchan << " channels x " << nseq
            << " sequences per gulp, " << tap.interval_s << " s tables to " << tap.path
            << (tap.record_dir.empty() ? "" : " and " + tap.record_dir) << ", runtime "
            << p_options["runtime"].as<int>() << " s";
  auto gen = get_pkt_gen_k<0>(p_options);
  VisTapSink<KernelTypeDefs::payload_u8_t> sink(tap, dev, nchan, nseq);
  raft::map m;
  m += *(gen.get()) >> sink;
  m.exe();
  LOG(INFO) << "vistap: done";
}

void RunEpic(int argc, char** argv) {
  using std::string_literals::operator""s;
  auto option_list = GetEpicOptions();
  LOG(INFO) << "Parsing options";
  auto options = option_list.parse(argc, argv);
  auto opt_valid = ValidateOptions(options);
  LOG(INFO) << "Validated options";

  if (opt_valid.value_or("none") != "none"s) {
    LOG(FATAL) << opt_valid.value();
  }

  if (options.count("help")) {
    std::cout << option_list.help() << std::endl;
    return;
  }

  if (options["printendpoints"].as<bool>()) {
    PrintStationEndPoints<LWA_SV>();
    return;
  }

  // Initialize the metrics exporter
  LOG(INFO) << "Initializing the metrics writer";
  PrometheusExporter::GetInstance(
      options["metrics_bind_addr"].as<std::string>(),
      options["disable_metrics"].as<bool>());

  InitInfoMetric(options);

  if (options["bench"].as<bool>() && !options["bench_full"].as<bool>()) {
    RunEpicBench(options);
    return;
  }

  if (options["vistap"].as<bool>()) {
    RunEpicVisTap(options);
    return;
  }
  
  LOG(INFO) << "Looking for GPUs";
  int num_gpus =
      options["ngpus"].as<int>() > 0 ? options["ngpus"].as<int>() : 1;
  raft::map m;

  LOG(INFO) << "Initializing EPIC on " << num_gpus << " GPU(s)";
  if (num_gpus == 3) {
    auto epic = EPIC<3>(options, &m);
    LOG(INFO) << "Running...";
    m.exe();
  } else if (num_gpus == 2) {
    auto epic = EPIC<2>(options, &m);
    LOG(INFO) << "Running...";
    m.exe();
  } else {
    auto epic = EPIC<1>(options, &m);
    LOG(INFO) << "Running...";
    m.exe();
  }
}

#endif  // SRC_RAFT_KERNELS_EPIC_EXECUTOR_HPP_
