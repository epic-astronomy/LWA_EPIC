#ifndef SRC_EX_PACKET_PROCESSOR_HPP_
#define SRC_EX_PACKET_PROCESSOR_HPP_

#include <arpa/inet.h>
#include <glog/logging.h>
#include <omp.h>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "constants.h"
#include "formats.h"
#include "helper_traits.hpp"
#include "hwy/cache_control.h"
#include "hwy/highway.h"
#include "math.h"

namespace hn = hwy::HWY_NAMESPACE;

using namespace std::string_literals;

template <typename Frmt, typename Dtype,
          template <class, class, PKT_DATA_ORDER> class Copier,
          PKT_DATA_ORDER Order = TIME_MAJOR>
class PacketProcessor : public Copier<Frmt, Dtype, Order> {
 public:
  using header_t = Frmt;
  using data_t = Dtype;
  static size_t nsrc;
  static constexpr int hdr_size = sizeof(Frmt);
  static constexpr int align_offset = AlignmentOffset<Frmt, Dtype>::value;
  inline static bool is_pkt_valid(Dtype* p_pkt, header_t& p_out_hdr,
                                  Dtype*& p_out_data, int p_nbytes);
  template <class Buffer>
  inline static void set_metadata(Buffer* p_mbuf, Frmt& p_hdr,
                                  uint64_t p_seq_start, uint64_t p_seq_end,
                                  uint64_t p_time_tag0);
  template <class Buffer>
  inline static void nullify_ill_sources(
      Buffer* p_mbuf, Frmt& p_hdr,
      std::vector<std::shared_ptr<std::atomic_ullong>>& p_pkt_stats,
      size_t p_exp_pkts_per_source);
};

template <typename Frmt, typename Dtype, PKT_DATA_ORDER Order>
class DefaultCopier {
 public:
  inline static void copy_to_buffer(
      const Frmt& p_hdr, Dtype* p_in_data, Dtype* p_out_data,
      std::vector<std::shared_ptr<std::atomic_ullong>>& p_pkt_stats, int p_tidx,
      int p_nseq_per_gulp = 0);
};

template <typename Frmt, typename Dtype, PKT_DATA_ORDER Order>
class AlignedCopier {
 public:
  inline static void copy_to_buffer(
      const Frmt& p_hdr, Dtype* p_in_data, Dtype* p_out_data,
      std::vector<std::shared_ptr<std::atomic_ullong>>& p_pkt_stats, int p_tidx,
      int p_nseq_per_gulp = 0);
};

template <PKT_DATA_ORDER Order>
class AlignedCopier<chips_hdr_type, uint8_t, Order> {
  static_assert(HWY_LANES(uint8_t) == 32,
                "Invalid lane size. Is the program compiled with AVX2?");

 public:
  inline static void copy_to_buffer(
      const chips_hdr_type& p_hdr, uint8_t* p_in_data, uint8_t* p_out_data,
      std::vector<std::shared_ptr<std::atomic_ullong>>& p_pkt_stats, int p_tidx,
      int p_nseq_per_gulp = 0);
};

template <PKT_DATA_ORDER Order>
class AlignedCopier<snap2_hdr_type, uint8_t, Order> {
  static_assert(HWY_LANES(uint8_t) == 32,
                "Invalid lane size. Is the program compiled with AVX2?");

 public:
  inline static void copy_to_buffer(
      const snap2_hdr_type& p_hdr, uint8_t* p_in_data, uint8_t* p_out_data,
      std::vector<std::shared_ptr<std::atomic_ullong>>& p_pkt_stats, int p_tidx,
      int p_nseq_per_gulp = 0);
};

/////////////////////////////////////////////////////

//////////////////////////////////////////////////////////
///////////////////////////// CHIPS //////////////////////
//////////////////////////////////////////////////////////

template <template <class, class, PKT_DATA_ORDER> class Copier,
          PKT_DATA_ORDER Order>
class PacketProcessor<chips_hdr_type, uint8_t, Copier, Order>
    : public Copier<chips_hdr_type, uint8_t, Order> {
 public:
  using header_t = chips_hdr_type;
  using data_t = uint8_t;
  static size_t nsrc;  // = size_t(NROACH_BOARDS);
  static constexpr int hdr_size = sizeof(header_t);
  static constexpr int align_offset = AlignmentOffset<header_t, data_t>::value;
  inline static bool is_pkt_valid(data_t* p_pkt, header_t& p_out_hdr,
                                  data_t*& p_out_data, int p_nbytes = 0);
  template <class Buffer>
  inline static void set_metadata(Buffer* p_mbuf, header_t& p_hdr,
                                  uint64_t p_seq_start, uint64_t p_seq_end,
                                  uint64_t p_time_tag0);
  template <class Buffer>
  inline static void nullify_ill_sources(
      Buffer* p_mbuf, header_t& p_hdr,
      std::vector<std::shared_ptr<std::atomic_ullong>>& p_pkt_stats,
      size_t p_exp_pkts_per_source);
  /// Index of the packet's source in the packet statistics
  inline static int source_index(const header_t& p_hdr) { return p_hdr.roach; }
  /// Sample weights (see the NDP processor); not implemented for CHIPS
  template <class Buffer>
  inline static void weigh_samples(
      Buffer*, header_t&, std::vector<std::shared_ptr<std::atomic_ullong>>&,
      const std::vector<uint64_t>&, size_t) {}
  // ~PacketProcessor(){
  //     DLOG(INFO)<<"D PktProcessor";
  // }
};

template <template <class, class, PKT_DATA_ORDER> class Copier,
          PKT_DATA_ORDER Order>
size_t PacketProcessor<chips_hdr_type, uint8_t, Copier, Order>::nsrc =
    size_t(NROACH_BOARDS);

template <template <class, class, PKT_DATA_ORDER> class Copier,
          PKT_DATA_ORDER Order>
bool PacketProcessor<chips_hdr_type, uint8_t, Copier, Order>::is_pkt_valid(
    uint8_t* p_pkt, chips_hdr_type& p_out_hdr, uint8_t*& p_out_data,
    int p_pkt_size) {
  if (p_pkt_size < sizeof(chips_hdr_type)) {
    return false;
  }
  int pld_size = p_pkt_size - sizeof(chips_hdr_type);
  const chips_hdr_type* pkt_hdr = (chips_hdr_type*)p_pkt;
  if (pld_size != pkt_hdr->nchan * CHIPS_NINPUTS_PER_CHANNEL) {
    return false;
  }
  p_out_data = p_pkt + sizeof(chips_hdr_type);
  p_out_hdr.seq = be64toh(pkt_hdr->seq);
  p_out_hdr.roach = (pkt_hdr->roach - 1);
  p_out_hdr.nchan = pkt_hdr->nchan;
  p_out_hdr.chan0 = ntohs(pkt_hdr->chan0);

  return (p_out_hdr.seq >= 0 && p_out_hdr.roach >= 0 &&
          p_out_hdr.roach < NROACH_BOARDS && p_out_hdr.chan0 >= 0);
}

template <PKT_DATA_ORDER Order>
void AlignedCopier<chips_hdr_type, uint8_t, Order>::copy_to_buffer(
    const chips_hdr_type& p_hdr, uint8_t* p_in_data, uint8_t* p_out_data,
    std::vector<std::shared_ptr<std::atomic_ullong>>& p_pkt_stats, int p_tidx,
    int p_nseq_per_gulp) {
  p_pkt_stats[p_hdr.roach]->fetch_add(1, std::memory_order_relaxed);
  hn::ScalableTag<uint8_t> tag8;
  hn::ScalableTag<uint8_t> _loadv;
  using v256_t = decltype(hn::Zero(tag8));
  assert((void("Misaligned input data ptr"),
          uint64_t(p_in_data) % HWY_LANES(uint8_t) == 0));
  assert((void("Misaligned output data ptr "),
          uint64_t(p_out_data) % HWY_LANES(uint8_t) == 0));

  // each vector holds 256 bits. Hence for each channel in a sequence
  // there will be NROACH_BOARD vectors
  auto vec_start = reinterpret_cast<v256_t*>(p_out_data);
  auto vec_pkt = reinterpret_cast<v256_t*>(p_in_data);

  int idx_multiplier = 1;
  if (Order == TIME_MAJOR) {
    // data dimensions = time, chan, ant, pol, real_imag
    vec_start = vec_start + p_tidx * NROACH_BOARDS * p_hdr.nchan + p_hdr.roach;
    idx_multiplier = NROACH_BOARDS;
  }
  if (Order == CHAN_MAJOR) {
    // data dimensions = chan, time, ant, pol, real_imag
    // this could be a potentially slower copy than TIME_MAJOR as the distance
    // between each vector access becomes larger by a factor of p_nseq_per_gulp
    vec_start = vec_start + p_tidx * NROACH_BOARDS + p_hdr.roach;
    idx_multiplier = NROACH_BOARDS * p_nseq_per_gulp;
  }

  std::atomic<bool> direction{true};
  int nsteps = 4;
  int chan = 0;
  //   if (!direction.load()) {
  //     nsteps = -nsteps;
  //     chan = p_hdr.nchan - 1;
  //   }
  // nsteps=4;
  // chan=0;
  int offset = (p_tidx * NROACH_BOARDS * p_hdr.nchan + p_hdr.roach) * 32;
  for (; chan < p_hdr.nchan && chan >= 0; chan += nsteps) {
    // for (auto chan = 0; chan < p_hdr.nchan; chan += 4) {
    if (chan < (p_hdr.nchan - std::abs(nsteps))) {
      // Bad. This will pollute the cache. However, there is also no way to
      // copy 64 bytes (cache line size) at once to generate movntdq
      // instructions (aka write-combining). So, let's prefetch.
      // Alternatively, we can use hn::Stream to copy with non-temporal hints,
      // but it may be too slow with the current copy patterns
      // hwy::Prefetch(vec_pkt + chan + nsteps);
    }
    // hn::Store(Load(_loadv, p_in_data + (chan)*32), tag8,
    //           p_out_data + chan * 32 * idx_multiplier + offset);
    *(vec_start + (chan)*idx_multiplier) = *(vec_pkt + chan);
    *(vec_start + (chan + 1) * idx_multiplier) = *(vec_pkt + chan + 1);
    *(vec_start + (chan + 2) * idx_multiplier) = *(vec_pkt + chan + 2);
    *(vec_start + (chan + 3) * idx_multiplier) = *(vec_pkt + chan + 3);
    // *(vec_start + (chan + 4) * idx_multiplier) = *(vec_pkt + chan + 4);
    // *(vec_start + (chan + 5) * idx_multiplier) = *(vec_pkt + chan + 5);
    // *(vec_start + (chan + 6) * idx_multiplier) = *(vec_pkt + chan + 6);
    // *(vec_start + (chan + 7) * idx_multiplier) = *(vec_pkt + chan + 7);
  }
  //   if (direction.load() && chan != (p_hdr.nchan - 1)) {
  //     while (chan == (p_hdr.nchan - 1)) {
  //       *(vec_start + (chan)*idx_multiplier) = *(vec_pkt + chan);
  //       ++chan;
  //     }
  //   }

  //   if (!direction.load() && chan != 0) {
  //     while (chan == 0) {
  //       *(vec_start + (chan)*idx_multiplier) = *(vec_pkt + chan);
  //       --chan;
  //     }
  //   }

  //   direction.store(!direction.load(), std::memory_order_relaxed);
}

template <template <class, class, PKT_DATA_ORDER> class Copier,
          PKT_DATA_ORDER Order>
template <class Buffer>
void PacketProcessor<chips_hdr_type, uint8_t, Copier, Order>::set_metadata(
    Buffer* p_mbuf, chips_hdr_type& p_hdr, uint64_t p_seq_start,
    uint64_t p_seq_end, uint64_t p_time_tag0) {
  auto& mref = p_mbuf->GetMetadataRef();
  mref["seq_start"] = p_seq_start;
  mref["seq_end"] = p_seq_end;
  int nseqs = p_seq_end - p_seq_start;
  mref["nseqs"] = nseqs;
  mref["gulp_len_ms"] = (p_seq_end - p_seq_start) * SAMPLING_LEN_uS * 1e-3;
  mref["nchan"] = p_hdr.nchan;
  mref["chan0"] = int64_t(p_hdr.chan0);  // to meet alignment requirements
  mref["data_order"] = (Order == TIME_MAJOR ? "t_maj"s : "c_maj"s);
  mref["nbytes"] =
      p_hdr.nchan * LWA_SV_NPOLS * nseqs * LWA_SV_NSTANDS * 1 /*bytes*/;
  mref["time_tag"] = p_time_tag0 + p_seq_start * SEQ_MULT_S;
}

template <template <class, class, PKT_DATA_ORDER> class Copier,
          PKT_DATA_ORDER Order>
template <class Buffer>
void PacketProcessor<chips_hdr_type, uint8_t, Copier, Order>::
    nullify_ill_sources(
        Buffer* p_mbuf, chips_hdr_type& p_hdr,
        std::vector<std::shared_ptr<std::atomic_ullong>>& p_pkt_stats,
        size_t p_exp_pkts_per_source) {
  hn::ScalableTag<uint8_t> tag8;
  auto zero_vec = Zero(tag8);
  using v256_t = decltype(zero_vec);
  auto vec_start = reinterpret_cast<v256_t*>(p_mbuf->GetDataPtr());
  auto& metadata = p_mbuf->GetMetadataRef();  // by reference

  size_t npkts = 0;

  static float allowed_pkt_drp_frac = ALLOWED_PKT_DROP * 0.01;
  int roach = 0;
  for (auto src_stat : p_pkt_stats) {
    if (*src_stat >= allowed_pkt_drp_frac * p_exp_pkts_per_source) {
      npkts += (*src_stat);
      continue;
    }
    if (Order == TIME_MAJOR) {
      for (auto tidx = 0; tidx < p_exp_pkts_per_source; ++tidx) {
        auto vec_data = vec_start + tidx * NROACH_BOARDS * p_hdr.nchan + roach;
        for (auto chan = 0; chan < p_hdr.nchan; ++chan) {
          *(vec_data + chan * NROACH_BOARDS) = zero_vec;
        }  // chan loop
      }    // time loop
    }      // t_maj

    if (Order == CHAN_MAJOR) {
      for (size_t tidx = 0; tidx < p_exp_pkts_per_source; ++tidx) {
        auto vec_data = vec_start + tidx * NROACH_BOARDS + roach;
        for (auto chan = 0; chan < p_hdr.nchan; ++chan) {
          *(vec_data + (chan)*NROACH_BOARDS * p_exp_pkts_per_source) = zero_vec;
        }  // chan loop
      }    // time loop
    }      // chan_maj
    ++roach;
    metadata["src" + std::to_string(roach) + "_valid_frac"] =
        (*src_stat) / double(p_exp_pkts_per_source);
  }

  metadata["data_quality"] =
      double(npkts) / double(p_exp_pkts_per_source * p_pkt_stats.size());
}

//////////////////////////////////////////////////////////
///////////////////////////// NDP ////////////////////////
//////////////////////////////////////////////////////////

// Each (board, channel block) pair is treated as a packet source so that every
// source sends one packet per sequence.
inline int Snap2SrcIdx(const snap2_hdr_type& p_hdr) {
  return (p_hdr.pol0 / CHIPS_NINPUTS_PER_CHANNEL) * NDP_CHAN_BLOCKS +
         p_hdr.chan_block_id;
}

// Only the central NDP_NCHAN_SELECT channels of a stream are kept in the gulp.
// Returns the stream channel that becomes the first channel of the gulp.
inline int Snap2SelChan0(const snap2_hdr_type& p_hdr) {
  return (p_hdr.nchan_tot - NDP_NCHAN_SELECT) / 2;
}

/**
 * Zero the samples of an NDP gulp whose packets are missing and record which
 * are valid (see PacketProcessor<snap2_hdr_type>::weigh_samples).
 * p_valid: bit t of word src * words + t / 64 is set if the packet of source
 * src = board * NDP_CHAN_BLOCKS + block at sequence t is valid; p_sel0: the
 * stream channel of the gulp's first channel; p_block_nchan: channels per
 * block. Writes valid_mask (p_valid as bytes) and valid_block to p_meta.
 */
template <PKT_DATA_ORDER Order, class Meta>
inline void ApplySampleWeights(uint8_t* p_data, const std::vector<uint64_t>& p_valid, size_t p_nseq,
                               int p_sel0, int p_block_nchan, Meta& p_meta) {
  constexpr size_t kLane = LWA_SV_INP_PER_CHAN / NROACH_BOARDS;  // bytes of one board
  const size_t words = (p_nseq + 63) / 64;
  const uint64_t last = p_nseq % 64 ? (uint64_t(1) << (p_nseq % 64)) - 1 : ~uint64_t(0);
  for (int src = 0; src < NROACH_BOARDS * NDP_CHAN_BLOCKS; ++src) {
    const uint64_t* m = p_valid.data() + size_t(src) * words;
    int nvalid = 0;
    for (size_t w = 0; w < words; ++w) nvalid += __builtin_popcountll(m[w] & (w + 1 == words ? last : ~uint64_t(0)));
    if (size_t(nvalid) == p_nseq) continue;
    const int board = src / NDP_CHAN_BLOCKS, block = src % NDP_CHAN_BLOCKS;
    const int chan_ofst = block * p_block_nchan - p_sel0;
    const int chan_beg = std::max(chan_ofst, 0);
    const int chan_end = std::min(chan_ofst + p_block_nchan, int(NDP_NCHAN_SELECT));
    for (size_t t = 0; t < p_nseq; ++t) {
      if ((m[t / 64] >> (t % 64)) & 1) continue;
      for (int chan = chan_beg; chan < chan_end; ++chan) {
        const size_t cell = Order == TIME_MAJOR ? (t * NDP_NCHAN_SELECT + chan) : (size_t(chan) * p_nseq + t);
        std::memset(p_data + (cell * NROACH_BOARDS + board) * kLane, 0, kLane);
      }
    }
  }
  std::string mask(reinterpret_cast<const char*>(p_valid.data()), p_valid.size() * sizeof(uint64_t));
  std::string blocks(NDP_NCHAN_SELECT, '\0');
  for (int c = 0; c < int(NDP_NCHAN_SELECT); ++c) {
    blocks[c] = char(std::min((p_sel0 + c) / p_block_nchan, int(NDP_CHAN_BLOCKS) - 1));
  }
  p_meta["valid_mask"] = std::move(mask);
  p_meta["valid_block"] = std::move(blocks);
}

template <template <class, class, PKT_DATA_ORDER> class Copier,
          PKT_DATA_ORDER Order>
class PacketProcessor<snap2_hdr_type, uint8_t, Copier, Order>
    : public Copier<snap2_hdr_type, uint8_t, Order> {
 public:
  using header_t = snap2_hdr_type;
  using data_t = uint8_t;
  static size_t nsrc;
  static constexpr int hdr_size = sizeof(header_t);
  static constexpr int align_offset = AlignmentOffset<header_t, data_t>::value;
  inline static bool is_pkt_valid(data_t* p_pkt, header_t& p_out_hdr,
                                  data_t*& p_out_data, int p_nbytes = 0);
  template <class Buffer>
  inline static void set_metadata(Buffer* p_mbuf, header_t& p_hdr,
                                  uint64_t p_seq_start, uint64_t p_seq_end,
                                  uint64_t p_time_tag0);
  template <class Buffer>
  inline static void nullify_ill_sources(
      Buffer* p_mbuf, header_t& p_hdr,
      std::vector<std::shared_ptr<std::atomic_ullong>>& p_pkt_stats,
      size_t p_exp_pkts_per_source);
  /// Index of the packet's source (board, channel block) in the statistics
  inline static int source_index(const header_t& p_hdr);
  /**
   * Sample weights: a sample is valid (weight 1) if its packet arrived, else
   * its slot is zeroed (the buffers are reused, so it would hold an earlier
   * gulp's voltages) and its weight is 0. A packet holds one sequence of one
   * board (16 stands) and one channel block. Recorded as metadata, for the
   * imager to correlate the weights like the voltages (FxImager):
   *   valid_mask   uint64 [board * NDP_CHAN_BLOCKS + block][(nseq + 63) / 64],
   *                bit t: the packet of sequence t arrived
   *   valid_block  uint8 [NDP_NCHAN_SELECT]: the block of each gulp channel
   * p_mask: bit t of word src * words + t / 64 is set if the packet of source
   * src at sequence t arrived. Run after nullify_ill_sources.
   */
  template <class Buffer>
  inline static void weigh_samples(
      Buffer* p_mbuf, header_t& p_hdr,
      std::vector<std::shared_ptr<std::atomic_ullong>>& p_pkt_stats,
      const std::vector<uint64_t>& p_mask, size_t p_exp_pkts_per_source);
};

template <template <class, class, PKT_DATA_ORDER> class Copier,
          PKT_DATA_ORDER Order>
size_t PacketProcessor<snap2_hdr_type, uint8_t, Copier, Order>::nsrc =
    size_t(NROACH_BOARDS * NDP_CHAN_BLOCKS);

/**
 * @brief Validate an NDP packet and decode its header to host byte order
 *
 */
template <template <class, class, PKT_DATA_ORDER> class Copier,
          PKT_DATA_ORDER Order>
bool PacketProcessor<snap2_hdr_type, uint8_t, Copier, Order>::is_pkt_valid(
    uint8_t* p_pkt, snap2_hdr_type& p_out_hdr, uint8_t*& p_out_data,
    int p_pkt_size) {
  if (p_pkt_size < static_cast<int>(sizeof(snap2_hdr_type))) {
    return false;
  }
  const snap2_hdr_type* pkt_hdr = (snap2_hdr_type*)p_pkt;
  p_out_hdr.seq = be64toh(pkt_hdr->seq);
  p_out_hdr.sync_time = be32toh(pkt_hdr->sync_time);
  p_out_hdr.npol = be16toh(pkt_hdr->npol);
  p_out_hdr.npol_tot = be16toh(pkt_hdr->npol_tot);
  p_out_hdr.nchan = be16toh(pkt_hdr->nchan);
  p_out_hdr.nchan_tot = be16toh(pkt_hdr->nchan_tot);
  p_out_hdr.chan_block_id = be32toh(pkt_hdr->chan_block_id);
  p_out_hdr.chan0 = be32toh(pkt_hdr->chan0);
  p_out_hdr.pol0 = be32toh(pkt_hdr->pol0);

  int pld_size = p_pkt_size - sizeof(snap2_hdr_type);
  if (p_out_hdr.npol != CHIPS_NINPUTS_PER_CHANNEL ||
      p_out_hdr.npol_tot != LWA_SV_INP_PER_CHAN ||
      pld_size != p_out_hdr.nchan * p_out_hdr.npol) {
    return false;
  }
  if (p_out_hdr.nchan_tot != p_out_hdr.nchan * NDP_CHAN_BLOCKS ||
      p_out_hdr.nchan_tot < NDP_NCHAN_SELECT ||
      p_out_hdr.chan_block_id >= NDP_CHAN_BLOCKS) {
    return false;
  }
  if (p_out_hdr.pol0 % CHIPS_NINPUTS_PER_CHANNEL != 0 ||
      p_out_hdr.pol0 / CHIPS_NINPUTS_PER_CHANNEL >= NROACH_BOARDS) {
    return false;
  }
  p_out_data = p_pkt + sizeof(snap2_hdr_type);
  return true;
}

template <PKT_DATA_ORDER Order>
void AlignedCopier<snap2_hdr_type, uint8_t, Order>::copy_to_buffer(
    const snap2_hdr_type& p_hdr, uint8_t* p_in_data, uint8_t* p_out_data,
    std::vector<std::shared_ptr<std::atomic_ullong>>& p_pkt_stats, int p_tidx,
    int p_nseq_per_gulp) {
  p_pkt_stats[Snap2SrcIdx(p_hdr)]->fetch_add(1, std::memory_order_relaxed);
  hn::ScalableTag<uint8_t> tag8;
  using v256_t = decltype(hn::Zero(tag8));
  assert((void("Misaligned input data ptr"),
          uint64_t(p_in_data) % HWY_LANES(uint8_t) == 0));
  assert((void("Misaligned output data ptr "),
          uint64_t(p_out_data) % HWY_LANES(uint8_t) == 0));

  // each vector holds the 32 inputs of a board for one channel
  auto vec_start = reinterpret_cast<v256_t*>(p_out_data);
  auto vec_pkt = reinterpret_cast<v256_t*>(p_in_data);
  int board = p_hdr.pol0 / CHIPS_NINPUTS_PER_CHANNEL;
  // position of this block's first channel in the stream
  int chan_ofst = p_hdr.chan_block_id * p_hdr.nchan;
  int sel0 = Snap2SelChan0(p_hdr);
  // stream channels in this packet that fall in the selected window
  int sel_beg = std::max(chan_ofst, sel0);
  int sel_end = std::min(chan_ofst + int(p_hdr.nchan), sel0 + NDP_NCHAN_SELECT);

  int idx_multiplier = 1;
  if (Order == TIME_MAJOR) {
    // data dimensions = time, chan, ant, pol, real_imag
    vec_start = vec_start + p_tidx * NDP_NCHAN_SELECT * NROACH_BOARDS + board;
    idx_multiplier = NROACH_BOARDS;
  }
  if (Order == CHAN_MAJOR) {
    // data dimensions = chan, time, ant, pol, real_imag
    vec_start = vec_start + p_tidx * NROACH_BOARDS + board;
    idx_multiplier = NROACH_BOARDS * p_nseq_per_gulp;
  }

  for (int chan = sel_beg; chan < sel_end; ++chan) {
    *(vec_start + (chan - sel0) * idx_multiplier) =
        *(vec_pkt + (chan - chan_ofst));
  }
}

template <template <class, class, PKT_DATA_ORDER> class Copier,
          PKT_DATA_ORDER Order>
template <class Buffer>
void PacketProcessor<snap2_hdr_type, uint8_t, Copier, Order>::set_metadata(
    Buffer* p_mbuf, snap2_hdr_type& p_hdr, uint64_t p_seq_start,
    uint64_t p_seq_end, uint64_t /*p_time_tag0*/) {
  auto& mref = p_mbuf->GetMetadataRef();
  mref["seq_start"] = p_seq_start;
  mref["seq_end"] = p_seq_end;
  int nseqs = p_seq_end - p_seq_start;
  mref["nseqs"] = nseqs;
  mref["gulp_len_ms"] = nseqs * SAMPLING_LEN_uS * 1e-3;
  mref["nchan"] = static_cast<uint8_t>(NDP_NCHAN_SELECT);
  // first selected channel. p_hdr.chan0 is the first channel of the packet.
  mref["chan0"] = int64_t(p_hdr.chan0 - p_hdr.chan_block_id * p_hdr.nchan +
                          Snap2SelChan0(p_hdr));
  mref["data_order"] = (Order == TIME_MAJOR ? "t_maj"s : "c_maj"s);
  mref["nbytes"] =
      NDP_NCHAN_SELECT * LWA_SV_NPOLS * nseqs * LWA_SV_NSTANDS * 1 /*bytes*/;
  // NDP sequences count from the Unix epoch, so p_time_tag0 is not needed
  mref["time_tag"] = p_seq_start * SEQ_MULT_S;
}

template <template <class, class, PKT_DATA_ORDER> class Copier,
          PKT_DATA_ORDER Order>
template <class Buffer>
void PacketProcessor<snap2_hdr_type, uint8_t, Copier, Order>::
    nullify_ill_sources(
        Buffer* p_mbuf, snap2_hdr_type& p_hdr,
        std::vector<std::shared_ptr<std::atomic_ullong>>& p_pkt_stats,
        size_t p_exp_pkts_per_source) {
  hn::ScalableTag<uint8_t> tag8;
  auto zero_vec = Zero(tag8);
  using v256_t = decltype(zero_vec);
  auto vec_start = reinterpret_cast<v256_t*>(p_mbuf->GetDataPtr());
  auto& metadata = p_mbuf->GetMetadataRef();  // by reference

  size_t npkts = 0;

  static float allowed_pkt_drp_frac = ALLOWED_PKT_DROP * 0.01;
  for (size_t src = 0; src < p_pkt_stats.size(); ++src) {
    auto nsrc_pkts = p_pkt_stats[src]->load();
    if (nsrc_pkts >= allowed_pkt_drp_frac * p_exp_pkts_per_source) {
      npkts += nsrc_pkts;
      continue;
    }
    int board = static_cast<int>(src / NDP_CHAN_BLOCKS);
    int block = static_cast<int>(src % NDP_CHAN_BLOCKS);
    // this source's channels in the gulp (i.e., the selected window)
    int chan_ofst = block * p_hdr.nchan - Snap2SelChan0(p_hdr);
    int chan_beg = std::max(chan_ofst, 0);
    int chan_end =
        std::min(chan_ofst + int(p_hdr.nchan), int(NDP_NCHAN_SELECT));
    for (size_t tidx = 0; tidx < p_exp_pkts_per_source; ++tidx) {
      for (auto chan = chan_beg; chan < chan_end; ++chan) {
        if (Order == TIME_MAJOR) {
          *(vec_start + (tidx * NDP_NCHAN_SELECT + chan) * NROACH_BOARDS +
            board) = zero_vec;
        }
        if (Order == CHAN_MAJOR) {
          *(vec_start + (chan * p_exp_pkts_per_source + tidx) * NROACH_BOARDS +
            board) = zero_vec;
        }
      }  // chan loop
    }    // time loop
    metadata["src" + std::to_string(src + 1) + "_valid_frac"] =
        nsrc_pkts / double(p_exp_pkts_per_source);
  }

  metadata["data_quality"] =
      double(npkts) / double(p_exp_pkts_per_source * p_pkt_stats.size());
}

template <template <class, class, PKT_DATA_ORDER> class Copier,
          PKT_DATA_ORDER Order>
int PacketProcessor<snap2_hdr_type, uint8_t, Copier, Order>::source_index(
    const snap2_hdr_type& p_hdr) {
  return Snap2SrcIdx(p_hdr);
}

template <template <class, class, PKT_DATA_ORDER> class Copier,
          PKT_DATA_ORDER Order>
template <class Buffer>
void PacketProcessor<snap2_hdr_type, uint8_t, Copier, Order>::weigh_samples(
    Buffer* p_mbuf, snap2_hdr_type& p_hdr,
    std::vector<std::shared_ptr<std::atomic_ullong>>& p_pkt_stats,
    const std::vector<uint64_t>& p_mask, size_t p_nseq) {
  const size_t words = (p_nseq + 63) / 64;
  static const float allowed = ALLOWED_PKT_DROP * 0.01;
  // the valid sequences of every source (none for a nullified one)
  std::vector<uint64_t> valid(size_t(nsrc) * words, 0);
  for (size_t src = 0; src < nsrc; ++src) {
    if (p_pkt_stats[src]->load() < allowed * p_nseq) continue;  // zeroed whole
    std::copy_n(p_mask.begin() + src * words, words, valid.begin() + src * words);
  }
  ApplySampleWeights<Order>(p_mbuf->GetDataPtr(), valid, p_nseq, Snap2SelChan0(p_hdr), int(p_hdr.nchan),
                            p_mbuf->GetMetadataRef());
}

void copy_tester() {
  int nchan = 12;
  int pkt_nitems = nchan * 32;
  int nseq = 2;
  auto pkt = hwy::AllocateAligned<uint8_t>(pkt_nitems);
  for (int i = 0; i < pkt_nitems; ++i) {
    pkt[i] = i / 32;
  }

  for (int i = 0; i < pkt_nitems; ++i) {
    std::cout << int(pkt.get()[i]) << " ";
  }
  std::cout << "\n\n\n";

  int gulp_nitems = pkt_nitems * NROACH_BOARDS * nseq;
  auto gulp = hwy::AllocateAligned<uint8_t>(gulp_nitems);

  chips_hdr_type hdr;
  hdr.gbe = 0;
  hdr.nchan = nchan;
  hdr.nsubband = 11;
  hdr.nsubband = 0;
  hdr.nroach = 16;
  hdr.chan0 = 1128;
  hdr.seq = 1;

  std::vector<std::shared_ptr<std::atomic_ullong>> m_valid_pkt_stats;
  for (int i = 0; i < NROACH_BOARDS; ++i) {
    m_valid_pkt_stats.push_back(std::make_shared<std::atomic_ullong>(0));
  }
  auto copier = AlignedCopier<chips_hdr_type, uint8_t, TIME_MAJOR>();
  for (int j = 0; j < nseq; ++j) {
    for (int roach = 0; roach < NROACH_BOARDS; ++roach) {
      hdr.roach = roach;
      copier.copy_to_buffer(hdr, pkt.get(), gulp.get(), m_valid_pkt_stats, j,
                            nseq);
    }
  }

  for (int i = 0; i < gulp_nitems; ++i) {
    std::cout << int(gulp.get()[i]) << " ";
  }
  std::cout << std::endl;
}

////////////////////////////////////////////////////
#endif  // SRC_EX_PACKET_PROCESSOR_HPP_
