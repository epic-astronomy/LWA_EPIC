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
#include "./accumulator.hpp"
#include "./chan_reducer.hpp"
#include "./correlator.hpp"
#include "./db_ingester.hpp"
#include "./disk_saver.hpp"
#include "./dummy_packet_gen.hpp"
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
  _DISK_SAVER = 8
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
  using ktype = std::unique_ptr<GulpGen_rft<vma_pkt_assembler>>;

  template <unsigned int _GpuId>
  static ktype get_kernel(const opt_t& options) {
    auto ip = options["addr"].as<std::vector<std::string>>();
    auto port = options["port"].as<std::vector<int>>();

    if ((ip.size() < (_GpuId + 1) || (port.size() < (_GpuId + 1)))) {
      LOG(FATAL) << "One IP address/port must be specified for each GPU";
    }
    LOG(INFO) << "Creating gulper";
    // using payload_t = typename ktype::payload_t;
    auto gulper = std::make_unique<vma_pkt_assembler>(ip[_GpuId], port[_GpuId]);

    return std::make_unique<GulpGen_rft<vma_pkt_assembler>>(
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
    auto gpu_ids = options["gpu_ids"].as<std::vector<int>>();
    auto correlator_options = MOFFCorrelatorDesc();
    correlator_options.device_id =
        gpu_ids.size() > 0 ? gpu_ids[_GpuId] : _GpuId;
    correlator_options.accum_time_ms = options["seq_accum"].as<int>();
    correlator_options.nseq_per_gulp = options["nts"].as<int>();
    correlator_options.nchan_out = options["channels"].as<int>();
    if (options["imagesize"].as<int>() == 64) {
      correlator_options.img_size = HALF;  // defaults to full
    }
    correlator_options.grid_res_deg = options["imageres"].as<float>();
    correlator_options.support_size = options["support"].as<int>();
    correlator_options.gcf_kernel_dim =
        std::sqrt(options["aeff"].as<std::vector<float>>()[_GpuId]) *
        10;  // radius of the kernel in decimeters
    correlator_options.kernel_oversampling_factor =
        options["kernel_oversample"].as<int>();
    correlator_options.use_bf16_accum = options["accum_16bit"].as<bool>();
    correlator_options.nstreams = options["nstreams"].as<int>();

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
    int imsize = options["imagesize"].as<int>();
    int im_nchan = options["channels"].as<int>();
    int chan_nbin = options["chan_nbin"].as<int>();
    return ktype(chan_nbin, imsize, imsize, im_nchan);
  }
};
using ChanReducer_kt = Kernel<_CHAN_REDUCER>::ktype;
template <unsigned int _GpuId>
auto& get_chan_reducer_k = Kernel<_CHAN_REDUCER>::get_kernel<_GpuId>;

template <>
struct Kernel<_PIX_EXTRACTOR> : KernelTypeDefs {
  using ktype = PixelExtractor<payload_float_t, pixel_pld_t, pixel_buf_t,
                               pixel_buf_config_t>;
  template <unsigned int _GpuId>
  static ktype get_kernel(const opt_t& options) {
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


    // fetch intial pixel indices;
    LOG(INFO)<<"Getting watch indices";
    auto initial_watch_indices =
        get_watch_indices(GetFirstSeqIdVma(ip[_GpuId], port[_GpuId]), grid_size,
                          grid_res, elev_limit_deg);
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
  static ktype get_kernel(const opt_t&) {
    return ktype();
  }
};
using IndexFetcher_kt = Kernel<_IDX_FETCHER>::ktype;
template <unsigned int _GpuId>
auto& get_index_fetcher_k = Kernel<_IDX_FETCHER>::get_kernel<_GpuId>;

template <>
struct Kernel<_DB_INGESTER> : KernelTypeDefs {
  using ktype = DBIngesterRft<pixel_pld_t>;
  template <unsigned int _GpuId>
  static ktype get_kernel(const opt_t&) {
    return ktype();
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
    int imsize = options["imagesize"].as<int>();
    int im_nchan = options["channels"].as<int>();
    int chan_nbin = options["chan_nbin"].as<int>();
    int reduced_nchan = im_nchan / chan_nbin;
    int im_naccum = options["nimg_accum"].as<int>();

    return ktype(imsize, imsize, reduced_nchan, im_naccum);
  }
};
using Accumulator_kt = Kernel<_ACCUMULATOR>::ktype;
template <unsigned int _GpuId>
auto& get_accumulator_k = Kernel<_ACCUMULATOR>::get_kernel<_GpuId>;

template <>
struct Kernel<_DISK_SAVER> : KernelTypeDefs {
  using ktype = DiskSaverRft<payload_float_t>;
  template <unsigned int _GpuId>
  static ktype get_kernel(const opt_t&) {
    return ktype(std::to_string(_GpuId));
  }
};
using DiskSaver_kt = Kernel<_DISK_SAVER>::ktype;
template <unsigned int _GpuId>
auto& get_disk_saver_k = Kernel<_DISK_SAVER>::get_kernel<_GpuId>;

#endif  // SRC_RAFT_KERNELS_KERNEL_TYPES_HPP_
