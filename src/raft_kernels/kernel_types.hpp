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

#ifndef SRC_RAFT_KERNELS_KERNEL_TYPES_HPP_
#define SRC_RAFT_KERNELS_KERNEL_TYPES_HPP_
#include <cmath>
#include <chrono>
#include <memory>
#include <raftmanip>
#include <string>
#include <vector>

#include "../ex/MOFF_correlator.hpp"
#include "../ex/buffer.hpp"
#include "../ex/helper_traits.hpp"
#include "../ex/lf_buf_mngr.hpp"
#include "../ex/option_parser.hpp"
#include "../ex/packet_assembler.hpp"
#include "../ex/station_desc.hpp"
#include "../ex/video_streaming.hpp"
#include "./accumulator.hpp"
#include "./chan_reducer.hpp"
#include "./correlator.hpp"
#include "./db_ingester.hpp"
#include "./disk_saver.hpp"
#include "./dummy_packet_gen.hpp"
#include "./epic_live_streamer.hpp"
#include "./index_fetcher.hpp"
#include "./packet_gen.hpp"
#include "./pixel_extractor.hpp"

enum EPICKernelID {
  _PKT_GEN = 0,
  _DUMMY_PACK_GEN = 1,
  _CORRELATOR = 2,
  _CHAN_REDUCER = 3,
  _PIX_EXTRACTOR = 4,
  _IDX_FETCHER = 5,
  _DB_INGESTER = 6,
  _ACCUMULATOR = 7,
  _DISK_SAVER = 8,
  _LIVE_STREAMER = 9
};

struct KernelTypeDefs {
  using lbuf_mngr_u8_t = LFBufMngr<AlignedBuffer<uint8_t>>;
  using lbuf_mngr_float_t = LFBufMngr<AlignedBuffer<float>>;
  using moffcorr_t = MOFFCorrelator<uint8_t, lbuf_mngr_float_t>;
  using mbuf_u8_t = typename lbuf_mngr_u8_t::mbuf_t;
  using mbuf_float_t = typename lbuf_mngr_float_t::mbuf_t;
  using payload_u8_t = Payload<mbuf_u8_t>;
  using payload_float_t = Payload<mbuf_float_t>;
  using pixel_buf_t = LFBufMngr<EpicPixelTableDataRows<float>>;
  using pixel_buf_config_t = typename EpicPixelTableDataRows<float>::config_t;
  using pixel_pld_t = Payload<typename pixel_buf_t::mbuf_t>;
  using opt_t = cxxopts::ParseResult;
};

template <EPICKernelID _kID>
struct Kernel : KernelTypeDefs {
  static_assert("Invalid EPIC kernel specified");
  using ktype = void;
  template <unsigned int _GpuId>
  ktype get_kernel();
};

template <>
struct Kernel<_PKT_GEN> : KernelTypeDefs {
  using pkt_assembler_t = verbs_pkt_assembler;
  //using pkt_assembler_t = vma_pkt_assembler;
  using ktype = std::unique_ptr<GulpGen_rft<pkt_assembler_t>>;

  template <unsigned int _GpuId>
  static ktype get_kernel(const opt_t& options) {
    VLOG(2) << "Creating Pkt gen";
    auto ip = options["addr"].as<std::vector<std::string>>();
    auto port = options["port"].as<std::vector<int>>();

    if ((ip.size() < (_GpuId + 1) || (port.size() < (_GpuId + 1)))) {
      LOG(FATAL) << "One IP address/port must be specified for each GPU";
    }
    LOG(INFO) << "Creating gulper";
    // using payload_t = typename ktype::payload_t;
    auto gulper = std::make_unique<pkt_assembler_t>(ip[_GpuId], port[_GpuId]);

    return std::make_unique<GulpGen_rft<pkt_assembler_t>>(
        gulper, options["runtime"].as<int>());
  }
};
using PktGen_kt = Kernel<_PKT_GEN>::ktype;
template <unsigned int _GpuId>
auto& get_pkt_gen_k = Kernel<_PKT_GEN>::get_kernel<_GpuId>;

template <>
struct Kernel<_DUMMY_PACK_GEN> : KernelTypeDefs {
  using ktype = DummyPktGen<payload_u8_t, LFBufMngr<AlignedBuffer<uint8_t>>>;
  template <unsigned int _GpuId>
  static ktype get_kernel(const opt_t&) {
    VLOG(2) << "Creating dummy Pkt gen";
    return ktype(2000);
  }
};
using DummyPktGen_kt = Kernel<_DUMMY_PACK_GEN>::ktype;
template <unsigned int _GpuId>
auto& get_dummy_pkt_gen_k = Kernel<_DUMMY_PACK_GEN>::get_kernel<_GpuId>;

template <>
struct Kernel<_CORRELATOR> : KernelTypeDefs {
  using ktype = CorrelatorRft<payload_u8_t, moffcorr_t>;
  template <unsigned int _GpuId>
  static ktype get_kernel(const opt_t& options) {
    VLOG(2) << "Creating Correlator";
    std::vector<int> gpu_ids;
    if (options.count("gpu_ids") > 0) {
      VLOG(2) << "inside";
      gpu_ids = options["gpu_ids"].as<std::vector<int>>();
    }
    VLOG(2) << "here";
    auto correlator_options = MOFFCorrelatorDesc();
    correlator_options.device_id =
        gpu_ids.size() > 0 ? gpu_ids[_GpuId] : _GpuId;
    correlator_options.accum_time_ms = options["seq_accum"].as<int>();
    correlator_options.nseq_per_gulp = options["nts"].as<int>();
    VLOG(2) << "here";
    correlator_options.nchan_out = options["channels"].as<int>();
    if (options["imagesize"].as<int>() == 64) {
      correlator_options.img_size = HALF;  // defaults to full
    } else if (options["imagesize"].as<int>() == 256) {
      correlator_options.img_size = DOUBLE;
    }
    correlator_options.use_fx = options["imager"].as<std::string>() == "fx";
    if (correlator_options.use_fx && !options["sky_shm"].as<std::string>().empty()) {
      correlator_options.sky_table_path = "/dev/shm/" + options["sky_shm"].as<std::string>() + "_table";
      correlator_options.sky_nbin = options["sky_nbin"].as<int>();
    }
    {
      // one image per buffer (floats): XX, YY, XY* for every channel; larger
      // images get fewer of these page-locked buffers (25 x 134 MB at 256)
      const int n = options["imagesize"].as<int>();
      size_t img = size_t(correlator_options.nchan_out) * n * n * 4;
      // the monitor's pixel spectra after the image (at most 65536 slots), the
      // peeled sources' peaks and the subtracted term
      // and the peaks of the sources peeled from them (5 a channel)
      if (!correlator_options.sky_table_path.empty()) img += size_t(correlator_options.nchan_out) * (2 * 65536 + 5);
      if (img > size_t(correlator_options.buf_size)) {
        correlator_options.buf_size = int(img);
        correlator_options.nbuffers = 25;
      }
    }
    VLOG(2) << "here";
    correlator_options.grid_res_deg = options["imageres"].as<float>();
    correlator_options.support_size = options["support"].as<int>();
    correlator_options.gcf_kernel_dim =
        std::sqrt(options["aeff"].as<std::vector<float>>()[_GpuId]) *
        10;  // radius of the kernel in decimeters
    correlator_options.kernel_oversampling_factor =
        options["kernel_oversample"].as<int>();
    correlator_options.use_bf16_accum = options["accum_16bit"].as<bool>();
    correlator_options.use_wproj = options["wproj"].as<bool>();
    correlator_options.is_remove_autocorr = options["remove_autocorr"].as<bool>();
    correlator_options.seq_chunk = options["seq_chunk"].as<int>();
    auto with_gpu = [&](std::string p) {
      auto pos = p.find("{gpu}");
      if (pos != std::string::npos) {
        p.replace(pos, 5, std::to_string(correlator_options.device_id));
      }
      return p;
    };
    correlator_options.gain_table_path = with_gpu(options["gain_table"].as<std::string>());
    correlator_options.proj_table_path = with_gpu(options["proj_table"].as<std::string>());
    correlator_options.impulse_sigma = options["impulse_sigma"].as<float>();
    correlator_options.caltap_nseq = options["caltap_nseq"].as<int>();
    correlator_options.caltap_gpu = options["caltap_gpu"].as<bool>();
    correlator_options.caltap_min_quality = options["caltap_min_quality"].as<float>();
    correlator_options.caltap_record_dir = options["caltap_record_dir"].as<std::string>();
    LOG_IF(FATAL, correlator_options.caltap_gpu && correlator_options.caltap_nseq > 0)
        << "--caltap_gpu and --caltap_nseq are alternatives";
    correlator_options.caltap_path = correlator_options.caltap_nseq > 0 || correlator_options.caltap_gpu
                                         ? with_gpu(options["caltap_out"].as<std::string>())
                                         : "";
    correlator_options.caltap_interval_s = options["caltap_interval"].as<float>();
    correlator_options.caltap_threads = options["caltap_threads"].as<int>();
    correlator_options.caltap_batch = options["caltap_batch"].as<int>();
    correlator_options.nstreams = options["nstreams"].as<int>();
    VLOG(2) << "Extracted options";
    auto corr_ptr = std::make_unique<MOFFCorrelator_t>(correlator_options);
    return ktype(&corr_ptr);
  }
};
using EPICCorrelator_kt = Kernel<_CORRELATOR>::ktype;
template <unsigned int _GpuId>
auto& get_epiccorr_k = Kernel<_CORRELATOR>::get_kernel<_GpuId>;

template <>
struct Kernel<_CHAN_REDUCER> : KernelTypeDefs {
  using ktype = ChanReducerRft<payload_float_t, lbuf_mngr_float_t>;
  template <unsigned int _GpuId>
  static ktype get_kernel(const opt_t& options) {
    VLOG(2) << "Creating chan reducer";
    int imsize = options["imagesize"].as<int>();
    int im_nchan = options["channels"].as<int>();
    int chan_nbin = options["chan_nbin"].as<int>();
    return ktype(chan_nbin, imsize, imsize, im_nchan,
                 options["sky_shm"].as<std::string>(),
                 options["sky_shm_slots"].as<int>(), options["sky_nbin"].as<int>());
  }
};
using ChanReducer_kt = Kernel<_CHAN_REDUCER>::ktype;
template <unsigned int _GpuId>
auto& get_chan_reducer_k = Kernel<_CHAN_REDUCER>::get_kernel<_GpuId>;

template <>
struct Kernel<_LIVE_STREAMER> : KernelTypeDefs {
  using ktype = EpicLiveStream<payload_float_t>;
  template <unsigned int _GpuId>
  static ktype get_kernel(const opt_t& options) {
    return ktype();
  }
};

using EpicLiveStream_kt = Kernel<_LIVE_STREAMER>::ktype;
template <unsigned int _GpuId>
auto& get_epic_live_stream_k = Kernel<_LIVE_STREAMER>::get_kernel<_GpuId>;

/// First sequence of the generated gulps (--bench_seq0, or now rounded to a
/// gulp), the same for every caller in the process
inline uint64_t BenchFirstSeq(const KernelTypeDefs::opt_t& p_options) {
  static const uint64_t seq0 = [&] {
    uint64_t s = p_options["bench_seq0"].as<uint64_t>();
    if (s == 0) {
      const int nseq = p_options["nts"].as<int>();
      double now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
      s = uint64_t(now / (SEQ_MULT_S / double(FS))) / nseq * nseq;
    }
    return s;
  }();
  return seq0;
}

template <>
struct Kernel<_PIX_EXTRACTOR> : KernelTypeDefs {
  using ktype = PixelExtractor<payload_float_t, pixel_pld_t, pixel_buf_t,
                               pixel_buf_config_t>;
  template <unsigned int _GpuId>
  static ktype get_kernel(const opt_t& options) {
    VLOG(2) << "Creating Pixel extractor";
    KernelTypeDefs::pixel_buf_config_t config;
    int imsize = options["imagesize"].as<int>();
    int im_nchan = options["channels"].as<int>();
    int chan_nbin = options["chan_nbin"].as<int>();
    int reduced_nchan = im_nchan / chan_nbin;
    int grid_size = options["imagesize"].as<int>();
    float grid_res = options["imageres"].as<float>();
    float elev_limit_deg = options["elev_limit_deg"].as<float>();
    auto ip = options["addr"].as<std::vector<std::string>>();
    auto port = options["port"].as<std::vector<int>>();
    auto watchdog_addr = options["watchdog_addr"].as<std::string>();

    // fetch intial pixel indices; generated gulps (--bench --bench_full)
    // give the first sequence without listening to the network
    LOG(INFO) << "Getting watch indices";
    const bool bench = options["bench"].as<bool>() && options["bench_full"].as<bool>();
    auto initial_watch_indices =
        GetWatchIndices(bench ? BenchFirstSeq(options) : GetFirstSeqIdVerbs(ip[_GpuId], port[_GpuId]), grid_size,
                          grid_res, elev_limit_deg, watchdog_addr);
    initial_watch_indices.print();
    config.nchan = reduced_nchan;
    config.ncoords = initial_watch_indices.m_ncoords;
    config.nsrcs = initial_watch_indices.nsrcs;
    config.kernel_dim = initial_watch_indices.m_kernel_dim;
    return ktype(config, initial_watch_indices, imsize, imsize, reduced_nchan);
  }
};
using PixelExtractor_kt = Kernel<_PIX_EXTRACTOR>::ktype;
template <unsigned int _GpuId>
auto& get_pixel_extractor_k = Kernel<_PIX_EXTRACTOR>::get_kernel<_GpuId>;

template <>
struct Kernel<_IDX_FETCHER> : KernelTypeDefs {
  using ktype = IndexFetcherRft;
  template <unsigned int _GpuId>
  static ktype get_kernel(const opt_t& options) {
    VLOG(2) << "Creating Index fetcher";
    int grid_size = options["imagesize"].as<int>();
    float grid_res = options["imageres"].as<float>();
    float elev_limit_deg = options["elev_limit_deg"].as<float>();
    auto watchdog_addr = options["watchdog_addr"].as<std::string>();
    return ktype(grid_size, grid_res, elev_limit_deg, watchdog_addr);
  }
};
using IndexFetcher_kt = Kernel<_IDX_FETCHER>::ktype;
template <unsigned int _GpuId>
auto& get_index_fetcher_k = Kernel<_IDX_FETCHER>::get_kernel<_GpuId>;

template <>
struct Kernel<_DB_INGESTER> : KernelTypeDefs {
  using ktype = DBIngesterRft<pixel_pld_t>;
  template <unsigned int _GpuId>
  static ktype get_kernel(const opt_t& options) {
    return ktype(options["epic_data_schema"].as<std::string>());
  }
};
using DBIngester_kt = Kernel<_DB_INGESTER>::ktype;
template <unsigned int _GpuId>
auto& get_db_ingester_k = Kernel<_DB_INGESTER>::get_kernel<_GpuId>;

template <>
struct Kernel<_ACCUMULATOR> : KernelTypeDefs {
  using ktype = AccumulatorRft<payload_float_t>;
  template <unsigned int _GpuId>
  static ktype get_kernel(const opt_t& options) {
    VLOG(2) << "Creating accumulator";
    int imsize = options["imagesize"].as<int>();
    int im_nchan = options["channels"].as<int>();
    int chan_nbin = options["chan_nbin"].as<int>();
    int reduced_nchan = im_nchan / chan_nbin;
    int im_naccum = options["nimg_accum"].as<int>();
    int im_min_accum = options["nimg_accum_min"].as<int>();

    return ktype(imsize, imsize, reduced_nchan, im_naccum, im_min_accum, options["accum_max_gap"].as<int>());
  }
};
using Accumulator_kt = Kernel<_ACCUMULATOR>::ktype;
template <unsigned int _GpuId>
auto& get_accumulator_k = Kernel<_ACCUMULATOR>::get_kernel<_GpuId>;

template <>
struct Kernel<_DISK_SAVER> : KernelTypeDefs {
  using ktype = DiskSaverRft<payload_float_t>;
  template <unsigned int _GpuId>
  static ktype get_kernel(const opt_t& options) {
    VLOG(2) << "Creating disk saver";
    return ktype(options["epic_data_schema"].as<std::string>(), options["out_dir"].as<std::string>());
  }
};
using DiskSaver_kt = Kernel<_DISK_SAVER>::ktype;
template <unsigned int _GpuId>
auto& get_disk_saver_k = Kernel<_DISK_SAVER>::get_kernel<_GpuId>;

std::unique_ptr<Streamer> GetStreamer(const KernelTypeDefs::opt_t& options) {
  int vid_size = options["video_size"].as<int>();
  int grid_size = options["imagesize"].as<int>();
  int seq_accum = options["seq_accum"].as<int>();
  float fps = 1000.f / seq_accum;
  std::string url = options["stream_url"].as<std::string>();
  std::string cmap = options["stream_cmap"].as<std::string>();

  return std::make_unique<Streamer>(fps, vid_size, vid_size, grid_size, fps,
                                    url, cmap, AV_LOG_ERROR,
                                    options["imageres"].as<float>());
}
#endif  // SRC_RAFT_KERNELS_KERNEL_TYPES_HPP_
