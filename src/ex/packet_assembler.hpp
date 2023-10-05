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

#ifndef SRC_EX_PACKET_ASSEMBLER_HPP_
#define SRC_EX_PACKET_ASSEMBLER_HPP_

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "./buffer.hpp"
#include "./constants.h"
#include "./exceptions.hpp"
#include "./helper_traits.hpp"
#include "./lf_buf_mngr.hpp"
#include "./metrics.hpp"
#include "./packet_processor.hpp"
#include "./packet_receiver.hpp"
#include "./py_funcs.hpp"
#include "./sockets.h"
// #include <chrono>

using namespace std::chrono;

/**
 * @brief Interface to sort and assemble the packets into gulps
 *
 * @tparam BufferMngr Type of the buffer manager
 * @tparam Receiver Type of receiver
 * @tparam PktProcessor Type of the packet processor
 */
template <class BufferMngr, class Receiver, class PktProcessor>
class PacketAssembler : public PktProcessor {
 public:
  using mbuf_t = typename BufferMngr::mbuf_t;
  using payload_t = Payload<mbuf_t>;
  using dtype = typename PktProcessor::data_t;

 protected:
  // using hdr_t = PktProcessor::hdr_t;
  std::unique_ptr<BufferMngr> m_buf_mngr;
  std::unique_ptr<Receiver> m_receiver;
  uint64_t m_seq_start{0};
  uint64_t m_seq_end{0};
  uint64_t m_time_tag0;
  size_t m_buf_ngulps;  // No. of gulps in the buffer manager
  size_t m_buf_size;    // size of each buffer
  size_t m_nseq_per_gulp;
  dtype* m_recent_data;
  typename PktProcessor::header_t m_recent_hdr;
  dtype* m_recent_pkt;
  std::vector<std::shared_ptr<std::atomic_ullong>> m_valid_pkt_stats;
  size_t m_n_valid_pkts;
  size_t m_n_valid_bytes;
  size_t m_min_pkt_limit;
  void m_reset_pkt_stats();
  void m_nudge_seq_start();  // to the nearest second
  bool m_last_pkt_available{false};

  unsigned int m_gauge_pktqual_id{0};

 public:
  PacketAssembler(std::string p_ip, int p_port, size_t p_nseq_per_gulp = 1000,
                  size_t p_ngulps = 20, size_t p_seq_size = SINGLE_SEQ_SIZE);
  payload_t get_gulp();
  size_t GetNumSeqPerGulp() { return m_nseq_per_gulp; }
  // ~PacketAssembler(){
  //     LOG(INFO)<<"D assembler";
  // }
};

template <typename BufferMngr, class Receiver, class PktProcessor>
PacketAssembler<BufferMngr, Receiver, PktProcessor>::PacketAssembler(
    std::string p_ip, int p_port, size_t p_nseq_per_gulp, size_t p_buf_ngulps,
    size_t p_seq_size)
    : m_buf_mngr(new BufferMngr(p_buf_ngulps, p_nseq_per_gulp * p_seq_size)),
      m_receiver(new Receiver()),
      m_nseq_per_gulp(p_nseq_per_gulp),
      m_buf_size(p_nseq_per_gulp * p_seq_size),
      m_buf_ngulps(p_buf_ngulps) {
  for (int i = 0; i < PktProcessor::nsrc; ++i) {
    m_valid_pkt_stats.push_back(std::make_shared<std::atomic_ullong>(0));
  }
  m_reset_pkt_stats();
  m_recent_hdr.seq = 0;
  m_seq_end = m_seq_start + m_nseq_per_gulp;
  VLOG(3) << "Setting address";
  m_receiver->set_address(p_ip, p_port);
  VLOG(3) << "Binding address";
  m_receiver->bind_socket();
  // std::cout << "initing receiver address\n";
  if (Receiver::type == VERBS) {
    VLOG(4) << AlignmentOffset<chips_hdr_type, uint8_t,
                               BF_VERBS_PAYLOAD_OFFSET>::value
            << "\n";
    m_receiver->init_receiver(AlignmentOffset<chips_hdr_type, uint8_t,
                                              BF_VERBS_PAYLOAD_OFFSET>::value);
  }
  m_min_pkt_limit = static_cast<float>(ALLOWED_PKT_DROP) * 0.01 *
                    PktProcessor::nsrc * m_nseq_per_gulp;
  m_time_tag0 = GetAdpTimeFromUnixEpoch() * FS;

  m_gauge_pktqual_id = PrometheusExporter::AddRuntimeSummaryLabel(
      {{"type", "data_quality"},
       {"units", "fraction"},
       {"kernel", "packet_assembler"}});
}

template <typename BufferMngr, class Receiver, class PktProcessor>
void PacketAssembler<BufferMngr, Receiver, PktProcessor>::m_reset_pkt_stats() {
  for (auto stats : m_valid_pkt_stats) {
    stats->exchange(0);
  }
  m_n_valid_pkts = 0;
  m_n_valid_bytes = 0;
}

template <typename BufferMngr, class Receiver, class PktProcessor>
void PacketAssembler<BufferMngr, Receiver, PktProcessor>::m_nudge_seq_start() {
  m_seq_start =
      size_t((m_seq_start - 1) / static_cast<double>(NSEQ_PER_SEC) + 2) *
      NSEQ_PER_SEC;
  m_seq_end = m_seq_start + m_nseq_per_gulp;
}

template <typename BufferMngr, class Receiver, class PktProcessor>
typename PacketAssembler<BufferMngr, Receiver, PktProcessor>::payload_t
PacketAssembler<BufferMngr, Receiver, PktProcessor>::get_gulp() {
  auto payload = m_buf_mngr->acquire_buf();
  auto mbuf = payload.get_mbuf();
  // payload.mbuf_shared_count();
  auto start = high_resolution_clock::now();
  auto stop = high_resolution_clock::now();
  int nbytes;
  int once = 0;
  int recvd_pkts = 0;
  VLOG(1) << "Generating a gulp";
  m_reset_pkt_stats();
  while (true) {
    if (!m_last_pkt_available) {  // fetch a new one
      start = high_resolution_clock::now();
      // VLOG(3) << "Receiving a packet";
      nbytes =
          m_receiver->recv_packet(m_recent_pkt, PktProcessor::align_offset);
      stop = high_resolution_clock::now();
      ++recvd_pkts;

      if (nbytes <= 0) {
        continue;
      }

      if (!PktProcessor::is_pkt_valid(m_recent_pkt, m_recent_hdr, m_recent_data,
                                      nbytes)) {
        VLOG(3) << "Bad header";
        continue;
      }
      if (m_recent_hdr.seq < m_seq_start) {
        m_last_pkt_available = false;
        continue;
      }
    }

    if (std::greater_equal<uint64_t>{}(m_recent_hdr.seq, m_seq_end)) {
      if (m_seq_start == 0) {  // packet assembling hasn't begun yet
        m_seq_start = m_recent_hdr.seq;
        VLOG(2) << "Nudging packet sequence to the nearest second";
        m_nudge_seq_start();
        continue;
      } else {
        VLOG(3) << "returning packet\n";
        VLOG(3) << "mvalid packets: " << m_n_valid_pkts << " 16000"
                << " " << recvd_pkts;
        VLOG(3) << m_seq_start << " " << m_seq_end << " " << m_recent_hdr.seq;
        m_last_pkt_available = true;
        VLOG(3) << "Nseqs: " << int(m_seq_start - m_seq_end);
        VLOG(3) << "data: " << int(mbuf->GetDataPtr()[0]) << " "
                << int(mbuf->GetDataPtr()[1]);

        if (m_recent_hdr.seq >= (m_seq_end + m_nseq_per_gulp)) {
          m_seq_start = std::ceil(static_cast<double>(m_recent_hdr.seq) /
                                  static_cast<double>(NSEQ_PER_SEC)) *
                        NSEQ_PER_SEC;  // m_recent_hdr.seq;
          m_seq_end = m_seq_start + m_nseq_per_gulp;
        } else {
          m_seq_start = m_seq_end;  // m_recent_hdr.seq;
          m_seq_end = m_seq_start + m_nseq_per_gulp;
        }

        if (m_n_valid_pkts < m_min_pkt_limit) {
          VLOG(3) << "returning null pkt";
          m_n_valid_pkts = 0;
          PrometheusExporter::ObserveRunTimeValue(
              m_gauge_pktqual_id,
              m_n_valid_pkts /
                  double(PktProcessor::nsrc*m_nseq_per_gulp) /*No data*/);

          return payload_t(nullptr);
        }

        PktProcessor::set_metadata(mbuf, m_recent_hdr, m_seq_start, m_seq_end,
                                   m_time_tag0);
        PktProcessor::nullify_ill_sources(mbuf, m_recent_hdr, m_valid_pkt_stats,
                                          m_nseq_per_gulp);

        m_n_valid_pkts = 0;
        // DLOG(INFO)<<"Returning payload";
        int count = 0;
        // for (auto i : m_valid_pkt_stats) {
        //   std::cout << "SRC: " << ++count << " " << *i << std::endl;
        // }
        PrometheusExporter::ObserveRunTimeValue(
            m_gauge_pktqual_id,
            std::get<double>(mbuf->GetMetadataRef()["data_quality"]));
        return payload;
      }
    }
    start = high_resolution_clock::now();

    PktProcessor::copy_to_buffer(
        m_recent_hdr, m_recent_data, mbuf->GetDataPtr(), m_valid_pkt_stats,
        m_recent_hdr.seq - m_seq_start, m_nseq_per_gulp);

    stop = high_resolution_clock::now();
    ++m_n_valid_pkts;
    m_last_pkt_available = false;
  }
}

// define
using default_buf_mngr_t = LFBufMngr<AlignedBuffer<uint8_t>>;
using float_buf_mngr_t = LFBufMngr<AlignedBuffer<float>>;
using default_receiver_t =
    VerbsReceiver<uint8_t, AlignedBuffer, MultiCastUDPSocket,
                  REG_COPY>;  // VMAReceiver<uint8_t, AlignedBuffer,
                              // MultiCastUDPSocket, REG_COPY>;

#ifdef _USE_VMA_
using vma_receiver_t =
    VMAReceiver<uint8_t, AlignedBuffer, MultiCastUDPSocket, REG_COPY>;
#endif
using verbs_receiver_t =
    VerbsReceiver<uint8_t, AlignedBuffer, MultiCastUDPSocket, REG_COPY>;
using default_pkt_processor_t =
    PacketProcessor<chips_hdr_type, uint8_t, AlignedCopier, CHAN_MAJOR>;

template class PacketAssembler<default_buf_mngr_t, default_receiver_t,
                               default_pkt_processor_t>;

using default_pkt_assembler =
    PacketAssembler<default_buf_mngr_t, default_receiver_t,
                    default_pkt_processor_t>;
using verbs_pkt_assembler =
    PacketAssembler<default_buf_mngr_t, verbs_receiver_t,
                    default_pkt_processor_t>;

#ifdef _USE_VMA_
using vma_pkt_assembler = PacketAssembler<default_buf_mngr_t, vma_receiver_t,
                                          default_pkt_processor_t>;
#endif

#endif  // SRC_EX_PACKET_ASSEMBLER_HPP_
