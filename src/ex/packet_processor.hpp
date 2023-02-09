#ifndef PACKET_PROCESSOR
#define PACKET_PROCESSOR

#include "constants.h"
#include "formats.h"
#include "helper_traits.hpp"
#include "hwy/cache_control.h"
#include "hwy/highway.h"
#include "math.h"
#include <arpa/inet.h>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <omp.h>
#include <string>
#include <thread>

namespace hn = hwy::HWY_NAMESPACE;

using namespace std::string_literals;

template<typename Frmt, typename Dtype, template<class, class, PKT_DATA_ORDER> class Copier, PKT_DATA_ORDER Order = TIME_MAJOR>
class PacketProcessor : public Copier<Frmt, Dtype, Order>
{
  public:
    using header_t = Frmt;
    using data_t = Dtype;
    static size_t nsrc;
    static constexpr int hdr_size = sizeof(Frmt);
    static constexpr int align_offset = alignment_offset<Frmt, Dtype>::value;
    inline static bool is_pkt_valid(Dtype* p_pkt, header_t& p_out_hdr, Dtype*& p_out_data, int p_nbytes);
    template<class Buffer>
    inline static void set_metadata(Buffer* p_mbuf, Frmt& p_hdr, uint64_t p_seq_start, uint64_t p_seq_end);
    template<class Buffer>
    inline static void nullify_ill_sources(Buffer* p_mbuf, Frmt& p_hdr, std::vector<std::shared_ptr<std::atomic_ullong>>& p_pkt_stats, size_t p_exp_pkts_per_source);
};

template<typename Frmt, typename Dtype, PKT_DATA_ORDER Order>
class DefaultCopier
{
  public:
    inline static void copy_to_buffer(const Frmt& p_hdr, Dtype* p_in_data, Dtype* p_out_data, std::vector<std::shared_ptr<std::atomic_ullong>>& p_pkt_stats, int p_tidx, int p_nseq_per_gulp = 0);
};

template<typename Frmt, typename Dtype, PKT_DATA_ORDER Order>
class AlignedCopier
{

  public:
    inline static void copy_to_buffer(const Frmt& p_hdr, Dtype* p_in_data, Dtype* p_out_data, std::vector<std::shared_ptr<std::atomic_ullong>>& p_pkt_stats, int p_tidx, int p_nseq_per_gulp = 0);
};

template<PKT_DATA_ORDER Order>
class AlignedCopier<chips_hdr_type, uint8_t, Order>
{
    static_assert(HWY_LANES(uint8_t) == 32, "Invalid lane size. Is the program compiled with AVX2?");

  public:
    inline static void copy_to_buffer(const chips_hdr_type& p_hdr, uint8_t* p_in_data, uint8_t* p_out_data, std::vector<std::shared_ptr<std::atomic_ullong>>& p_pkt_stats, int p_tidx, int p_nseq_per_gulp = 0);
};

/////////////////////////////////////////////////////

//////////////////////////////////////////////////////////
///////////////////////////// CHIPS //////////////////////
//////////////////////////////////////////////////////////

template<template<class, class, PKT_DATA_ORDER> class Copier, PKT_DATA_ORDER Order>
class PacketProcessor<chips_hdr_type, uint8_t, Copier, Order> : public Copier<chips_hdr_type, uint8_t, Order>
{
  public:
    using header_t = chips_hdr_type;
    using data_t = uint8_t;
    static size_t nsrc; // = size_t(NROACH_BOARDS);
    static constexpr int hdr_size = sizeof(header_t);
    static constexpr int align_offset = alignment_offset<header_t, data_t>::value;
    inline static bool is_pkt_valid(data_t* p_pkt, header_t& p_out_hdr, data_t*& p_out_data, int p_nbytes = 0);
    template<class Buffer>
    inline static void set_metadata(Buffer* p_mbuf, header_t& p_hdr, uint64_t p_seq_start, uint64_t p_seq_end);
    template<class Buffer>
    inline static void nullify_ill_sources(Buffer* p_mbuf, header_t& p_hdr, std::vector<std::shared_ptr<std::atomic_ullong>>& p_pkt_stats, size_t p_exp_pkts_per_source);
};

template<template<class, class, PKT_DATA_ORDER> class Copier, PKT_DATA_ORDER Order>
size_t PacketProcessor<chips_hdr_type, uint8_t, Copier, Order>::nsrc = size_t(NROACH_BOARDS);

template<template<class, class, PKT_DATA_ORDER> class Copier, PKT_DATA_ORDER Order>
bool
PacketProcessor<chips_hdr_type, uint8_t, Copier, Order>::is_pkt_valid(
  uint8_t* p_pkt,
  chips_hdr_type& p_out_hdr,
  uint8_t*& p_out_data,
  int p_pkt_size)
{
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

    return (p_out_hdr.seq >= 0 &&
            p_out_hdr.roach >= 0 && p_out_hdr.roach < NROACH_BOARDS &&
            p_out_hdr.chan0 >= 0);
}

template<PKT_DATA_ORDER Order>
void
AlignedCopier<chips_hdr_type, uint8_t, Order>::copy_to_buffer(const chips_hdr_type& p_hdr, uint8_t* p_in_data, uint8_t* p_out_data, std::vector<std::shared_ptr<std::atomic_ullong>>& p_pkt_stats, int p_tidx, int p_nseq_per_gulp)
{
    p_pkt_stats[p_hdr.roach]->fetch_add(1, std::memory_order_relaxed);
    hn::ScalableTag<uint8_t> tag8;
    using v256_t = decltype(hn::Zero(tag8));
    assert((void("Misaligned input data ptr"), uint64_t(p_in_data) % HWY_LANES(uint8_t) == 0));
    assert((void("Misaligned output data ptr "), uint64_t(p_out_data) % HWY_LANES(uint8_t) == 0));

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
        idx_multiplier = int(NROACH_BOARDS) * p_nseq_per_gulp;
    }

    std::atomic<bool> direction{ true };
    int nsteps = 8;
    int chan = 0;
    if (!direction.load()) {
        nsteps = -nsteps;
        chan = p_hdr.nchan - 1;
    }
    // nsteps=4;
    // chan=0;
    for (; chan < p_hdr.nchan && chan >= 0; chan += nsteps) {
        // for (auto chan = 0; chan < p_hdr.nchan; chan += 4) {
        if (chan < (p_hdr.nchan - std::abs(nsteps))) {
            // Bad. This will pollute the cache. However, there is also no way to
            // copy 64 bytes (cache line size) at once to generate movntdq
            // instructions (aka write-combining). So, let's prefetch.
            // Alternatively, we can use hn::Stream to copy with non-temporal hints,
            // but it may be too slow with the current copy patterns
            hwy::Prefetch(vec_pkt + chan + nsteps);
        }

        *(vec_start + (chan)*idx_multiplier) = *(vec_pkt + chan);
        *(vec_start + (chan + 1) * idx_multiplier) = *(vec_pkt + chan + 1);
        *(vec_start + (chan + 2) * idx_multiplier) = *(vec_pkt + chan + 2);
        *(vec_start + (chan + 3) * idx_multiplier) = *(vec_pkt + chan + 3);
        *(vec_start + (chan + 4) * idx_multiplier) = *(vec_pkt + chan + 4);
        *(vec_start + (chan + 5) * idx_multiplier) = *(vec_pkt + chan + 5);
        *(vec_start + (chan + 6) * idx_multiplier) = *(vec_pkt + chan + 6);
        *(vec_start + (chan + 7) * idx_multiplier) = *(vec_pkt + chan + 7);
    }
    if (direction.load() && chan != p_hdr.nchan - 1) {
        while (chan == p_hdr.nchan - 1) {
            *(vec_start + (chan)*idx_multiplier) = *(vec_pkt + chan);
            ++chan;
        }
    }

    if (!direction.load() && chan != 0) {
        while (chan == 0) {
            *(vec_start + (chan)*idx_multiplier) = *(vec_pkt + chan);
            --chan;
        }
    }

    direction.store(!direction.load(), std::memory_order_relaxed);
}

template<template<class, class, PKT_DATA_ORDER> class Copier, PKT_DATA_ORDER Order>
template<class Buffer>
void
PacketProcessor<chips_hdr_type, uint8_t, Copier, Order>::set_metadata(Buffer* p_mbuf, chips_hdr_type& p_hdr, uint64_t p_seq_start, uint64_t p_seq_end)
{
    auto& mref = p_mbuf->get_metadataref();
    mref["seq_start"] = p_seq_start;
    mref["seq_end"] = p_seq_end;
    mref["nchan"] = p_hdr.nchan;
    mref["chan0"] = int64_t(p_hdr.chan0); // to meet alignment requirements
    mref["data_order"] = (Order == TIME_MAJOR ? "t_maj"s : "c_maj"s);
}

template<template<class, class, PKT_DATA_ORDER> class Copier, PKT_DATA_ORDER Order>
template<class Buffer>
void
PacketProcessor<chips_hdr_type, uint8_t, Copier, Order>::nullify_ill_sources(Buffer* p_mbuf, chips_hdr_type& p_hdr, std::vector<std::shared_ptr<std::atomic_ullong>>& p_pkt_stats, size_t p_exp_pkts_per_source)
{
    hn::ScalableTag<uint8_t> tag8;
    auto zero_vec = Zero(tag8);
    using v256_t = decltype(zero_vec);
    auto vec_start = reinterpret_cast<v256_t*>(p_mbuf->get_data_ptr());
    auto& metadata = p_mbuf->get_metadataref(); // by reference

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
                } // chan loop
            }     // time loop
        }         // t_maj

        if (Order == CHAN_MAJOR) {
            for (size_t tidx = 0; tidx < p_exp_pkts_per_source; ++tidx) {
                auto vec_data = vec_start + tidx * NROACH_BOARDS + roach;
                for (auto chan = 0; chan < p_hdr.nchan; ++chan) {
                    *(vec_data + (chan)*NROACH_BOARDS * p_exp_pkts_per_source) = zero_vec;
                } // chan loop
            }     // time loop
        }         // chan_maj
        ++roach;
        metadata["src" + std::to_string(roach) + "_valid_frac"] = (*src_stat) / double(p_exp_pkts_per_source);
    }

    metadata["data_quality"] = double(npkts) / double(p_exp_pkts_per_source * p_pkt_stats.size());
}

////////////////////////////////////////////////////
#endif // PACKET_PROCESSOR