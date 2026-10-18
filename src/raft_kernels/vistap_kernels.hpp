// Copyright (c) 2026 The EPIC++ authors
//
// `epic++ --vistap`: the F-Engine receiver (ibverbs packet assembler, as in
// production) feeding the tensor-core correlator tap only. Every gulp is copied
// to the GPU and correlated (tcc_tap.h); every --caltap_interval the integrated
// XX/YY visibilities are published to --caltap_out and, with
// --caltap_record_dir, also written to one file per table. No imaging, no
// database, no video. Runs for --runtime seconds (the packet generator's timed
// mode) and exits.
#ifndef SRC_RAFT_KERNELS_VISTAP_KERNELS_HPP_
#define SRC_RAFT_KERNELS_VISTAP_KERNELS_HPP_

#include <cuda_runtime_api.h>
#include <glog/logging.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <raft>
#include <raftio>
#include <variant>

#include "../ex/constants.h"
#include "../ex/tcc_tap.h"

template <class Payload>
class VisTapSink : public raft::kernel {
 public:
  VisTapSink(TccTapOptions p_opts, int p_device, int p_nchan, int p_nseq)
      : raft::kernel(), m_device(p_device), m_nchan(p_nchan), m_nseq(p_nseq) {
    input.addPort<Payload>("gulp");
    cudaSetDevice(m_device);
    m_bytes = size_t(m_nchan) * m_nseq * LWA_SV_INP_PER_CHAN;
    LOG_IF(FATAL, cudaMalloc(&m_d_gulp, m_bytes) != cudaSuccess) << "vistap: cudaMalloc";
    LOG_IF(FATAL, cudaStreamCreate(&m_stream) != cudaSuccess) << "vistap: stream";
    m_tap = std::make_unique<TccTap>(p_opts, m_device, m_nchan, m_nseq, 1);
    m_t0 = std::chrono::steady_clock::now();
  }
  ~VisTapSink() {
    cudaSetDevice(m_device);
    cudaStreamSynchronize(m_stream);
    m_tap.reset();
    cudaFree(m_d_gulp);
    cudaStreamDestroy(m_stream);
    LOG(INFO) << "vistap: " << m_ngulps << " gulps received, " << m_ncorr << " correlated, " << m_nbad
              << " skipped (shape or quality)";
  }

  raft::kstatus run() override {
    Payload pld;
    input["gulp"].pop(pld);
    if (!pld) return raft::proceed;
    auto& meta = pld.get_mbuf()->GetMetadataRef();
    const uint64_t seq = std::get<uint64_t>(meta["seq_start"]);
    const int64_t chan0 = std::get<int64_t>(meta["chan0"]);
    const int nchan = int(std::get<uint8_t>(meta["nchan"]));
    const int nseq = std::get<int>(meta["nseqs"]);
    double q = 1.0;
    if (auto it = meta.find("data_quality"); it != meta.end()) {
      if (auto* v = std::get_if<double>(&it->second)) q = *v;
    }
    ++m_ngulps;
    m_qsum += q;
    cudaSetDevice(m_device);
    if (nchan >= m_nchan && nseq == m_nseq && m_tap->Arm(seq, chan0, nchan, nseq, float(q))) {
      m_tap->BeforeCopy(m_stream);
      cudaMemcpyAsync(m_d_gulp, pld.get_mbuf()->GetDataPtr(), m_bytes, cudaMemcpyHostToDevice, m_stream);
      m_tap->AfterCopy(0, m_stream);
      m_tap->Launch(m_d_gulp);
      cudaStreamSynchronize(m_stream);  // the host gulp buffer is released when pld goes out of scope
      ++m_ncorr;
    } else {
      ++m_nbad;
    }
    if (m_ngulps % 240 == 0) {
      const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - m_t0).count();
      LOG(INFO) << "vistap: " << m_ngulps << " gulps in " << el << " s (" << m_ngulps / el
                << " per s; real time is " << 1.0 / (m_nseq * SAMPLING_LEN_uS * 1e-6) << "), correlated " << m_ncorr
                << ", skipped " << m_nbad << ", mean data quality " << m_qsum / m_ngulps << ", tables "
                << m_tap->Version() << ", chan0 " << chan0;
    }
    return raft::proceed;
  }

 private:
  int m_device, m_nchan, m_nseq;
  size_t m_bytes{0};
  uint8_t* m_d_gulp{nullptr};
  cudaStream_t m_stream{nullptr};
  std::unique_ptr<TccTap> m_tap;
  int64_t m_ngulps{0}, m_ncorr{0}, m_nbad{0};
  double m_qsum{0};
  std::chrono::steady_clock::time_point m_t0;
};

#endif  // SRC_RAFT_KERNELS_VISTAP_KERNELS_HPP_
