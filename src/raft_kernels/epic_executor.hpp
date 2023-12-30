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

#include "../ex/metrics.hpp"
#include "../ex/station_desc.hpp"
#include "./kernel_types.hpp"
// #include "./epic_live_streamer.hpp"

template <unsigned int _GpuId>
class EPICKernels {
 private:
  std::unique_ptr<Streamer> m_streamer;
  bool m_is_offline{false};
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
    // use an additional offset of 2 for the runtime
    constexpr unsigned int cpu_ofst = (_GpuId)*m_nkernels + 2;
    // ensure the cpu ID is always non-zero.
    // Setting it to zero causes instability
    if (m_is_offline) {
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
  }

  void InitMap(const KernelTypeDefs::opt_t& p_options) {
    auto& m = *m_map;
    // m += m_dpkt_gen >> m_correlator >> m_disk_saver;
    if (m_is_offline) {
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
    if (!m_is_offline) {
      m_pkt_gen = std::move(get_pkt_gen_k<_GpuId>(p_options));
    }
    // m_disk_saver.SetStreamer(m_streamer.get());
    m_live_streamer.SetStreamer(m_streamer.get());

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
