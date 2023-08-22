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

#ifndef SRC_RAFT_KERNELS_DUMMY_PACKET_GEN_HPP_
#define SRC_RAFT_KERNELS_DUMMY_PACKET_GEN_HPP_
#include <glog/logging.h>

#include <chrono>
#include <memory>
#include <raft>
#include <raftio>
#include <string>

#include "../ex/buffer.hpp"
#include "../ex/constants.h"
#include "../ex/py_funcs.hpp"
#include "../ex/types.hpp"

template <class Payload, class BufferMngr>
class DummyPktGen : public raft::kernel {
 private:
  size_t m_n_pkts{3};
  const int m_ngulps{20};
  const int m_ngulps_per_seq{1000};
  const int m_nchan_in{128};
  const int m_chan0{1128 + 600};  // 1128+600
  uint64_t m_time_from_unix_epoch_s{0};
  uint64_t m_time_tag0{0};
  std::unique_ptr<BufferMngr> m_buf_mngr{nullptr};

 public:
  DummyPktGen(size_t p_n_pkts = 1,
                std::string utcstart = "2023_06_19T00_00_00")
      : raft::kernel(), m_n_pkts(p_n_pkts) {
    VLOG(3) << "Dummy pkt constructor";
    m_buf_mngr = std::make_unique<BufferMngr>(
        m_ngulps, m_ngulps_per_seq * SINGLE_SEQ_SIZE);
    output.addPort<Payload>("gulp");

    if (utcstart == "") {
      m_time_from_unix_epoch_s = get_ADP_time_from_unix_epoch();
    } else {
      m_time_from_unix_epoch_s = get_time_from_unix_epoch(utcstart);
    }
    m_time_tag0 = m_time_from_unix_epoch_s * FS;
  }

  raft::kstatus run() override {
    using std::string_literals::operator""s;
    for (size_t i = 0; i < m_n_pkts; ++i) {
      VLOG(3) << "Generating a gulp";
      auto pld = m_buf_mngr.get()->acquire_buf();
      LOG_IF(FATAL, !static_cast<bool>(pld)) << "Empty buffer in packet gen";

      auto start = std::chrono::high_resolution_clock::now();
      // get_40ms_gulp(pld.get_mbuf()->get_data_ptr());
      VLOG(3) << "Gulp gen duration: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - start)
                     .count();
      LOG(INFO) << "Gulp id: " << i;
      std::this_thread::sleep_for(std::chrono::milliseconds(40));

      VLOG(3)
          << "Sending gulp at: "
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::high_resolution_clock::now().time_since_epoch())
                 .count();

      auto& mref = pld.get_mbuf()->get_metadataref();
      uint64_t seq_start = 329008696996015680;
      mref["seq_start"] = seq_start;
      mref["time_tag"] =
          m_time_tag0 + std::get<uint64_t>(mref["seq_start"]) * SEQ_MULT_S;
      mref["seq_end"] = uint64_t(m_ngulps_per_seq + seq_start);
      int nseqs = m_ngulps_per_seq;
      mref["nseqs"] = nseqs;
      mref["gulp_len_ms"] = (m_ngulps_per_seq)*SAMPLING_LEN_uS * 1e3;
      mref["nchan"] = uint8_t(m_nchan_in);
      mref["chan0"] = int64_t(m_chan0);  // to meet alignment requirements
      mref["data_order"] = "t_maj"s;
      mref["nbytes"] =
          m_nchan_in * LWA_SV_NPOLS * nseqs * LWA_SV_NSTANDS * 1 /*bytes*/;

      output["gulp"].push(pld);
    }
    VLOG(2) << "Stopping gulp gen";
    return raft::stop;
  }
};

#endif  // SRC_RAFT_KERNELS_DUMMY_PACKET_GEN_HPP_
