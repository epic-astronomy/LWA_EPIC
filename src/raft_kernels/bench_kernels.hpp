// Copyright (c) 2026 The EPIC++ authors
//
// Kernels for `epic++ --bench`: the imager (CorrelatorRft, including the
// calibration tap and the gain table) driven by a gulp generator instead of
// the F-engine, with the images going to a sink that measures their cadence.
// No packet capture, database, disk or video output is involved.
//
//   BenchGulpGen  plays gulps from a file of simulated F-engine data
//                 ([gulp][chan][seq][stand][pol] CNib bytes, CHAN_MAJOR) or
//                 sends empty (zero) gulps. On every pass through the file the
//                 sequences of each channel are rotated by --bench_rotate, so
//                 the calibration tap sees new samples. Runs as fast as the
//                 pipeline accepts gulps, or paced at the real-time rate.
//   BenchSink     consumes images; reports the interval between images and
//                 the gain-table version; optionally dumps images to disk.
#ifndef SRC_RAFT_KERNELS_BENCH_KERNELS_HPP_
#define SRC_RAFT_KERNELS_BENCH_KERNELS_HPP_

#include <fcntl.h>
#include <glog/logging.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <raft>
#include <raftio>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "../ex/constants.h"

struct BenchOptions {
  std::string data_path;      // empty: zero gulps
  int64_t chan0{3104};
  int nchan{128};
  int nseq{1000};
  uint64_t seq0{0};           // first F-engine sequence (counts from the Unix epoch)
  int64_t ngulps{0};          // 0: run until stopped
  bool realtime{false};       // pace gulps at the F-engine rate
  int rotate{50};             // per-pass rotation of each channel's sequences
  std::string dump_dir;       // empty: no image dumps
  int dump_every{0};
  int report_every{120};      // images per statistics line
};

namespace bench_detail {
inline void Summarise(std::vector<double>& v, double* mean, double* p50, double* p99, double* mx) {
  if (v.empty()) {
    *mean = *p50 = *p99 = *mx = 0;
    return;
  }
  std::vector<double> s(v);
  std::sort(s.begin(), s.end());
  double sum = 0;
  for (double x : s) sum += x;
  *mean = sum / s.size();
  *p50 = s[s.size() / 2];
  *p99 = s[std::min(s.size() - 1, size_t(0.99 * s.size()))];
  *mx = s.back();
}
}  // namespace bench_detail

template <class Payload, class BufferMngr>
class BenchGulpGen : public raft::kernel {
 public:
  explicit BenchGulpGen(BenchOptions p_opts) : raft::kernel(), m_opts(std::move(p_opts)) {
    m_gulp_bytes = size_t(m_opts.nchan) * m_opts.nseq * LWA_SV_INP_PER_CHAN;
    m_buf_mngr = std::make_unique<BufferMngr>(20, m_gulp_bytes);
    output.addPort<Payload>("gulp");
    if (!m_opts.data_path.empty()) {
      int fd = ::open(m_opts.data_path.c_str(), O_RDONLY);
      LOG_IF(FATAL, fd < 0) << "bench: cannot open " << m_opts.data_path;
      struct stat st;
      fstat(fd, &st);
      m_nfile = st.st_size / m_gulp_bytes;
      LOG_IF(FATAL, m_nfile == 0) << "bench: " << m_opts.data_path << " holds no complete gulp";
      m_data = static_cast<const uint8_t*>(mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0));
      ::close(fd);
      m_map_bytes = st.st_size;
      LOG(INFO) << "bench: " << m_nfile << " simulated gulps from " << m_opts.data_path;
    } else {
      LOG(INFO) << "bench: empty gulps";
    }
  }
  ~BenchGulpGen() {
    if (m_data) munmap(const_cast<uint8_t*>(m_data), m_map_bytes);
  }

  raft::kstatus run() override {
    using clock = std::chrono::steady_clock;
    const double gulp_s = m_opts.nseq * SAMPLING_LEN_uS * 1e-6;
    auto t_start = clock::now();
    auto t_last = t_start;
    std::vector<double> wait_ms, interval_ms;
    for (int64_t i = 0; m_opts.ngulps <= 0 || i < m_opts.ngulps; ++i) {
      auto t0 = clock::now();
      Payload pld = m_buf_mngr->acquire_buf();
      while (!pld) {  // backpressure: every buffer is still in the pipeline
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        pld = m_buf_mngr->acquire_buf();
      }
      wait_ms.push_back(std::chrono::duration<double, std::milli>(clock::now() - t0).count());
      uint8_t* dst = pld.get_mbuf()->GetDataPtr();
      if (m_data) {
        const uint8_t* src = m_data + size_t(i % m_nfile) * m_gulp_bytes;
        const int shift = int((i / m_nfile) * m_opts.rotate % m_opts.nseq);
        const size_t per_seq = LWA_SV_INP_PER_CHAN;
        for (int c = 0; c < m_opts.nchan; ++c) {
          const uint8_t* s = src + size_t(c) * m_opts.nseq * per_seq;
          uint8_t* d = dst + size_t(c) * m_opts.nseq * per_seq;
          std::memcpy(d, s + size_t(shift) * per_seq, size_t(m_opts.nseq - shift) * per_seq);
          if (shift) std::memcpy(d + size_t(m_opts.nseq - shift) * per_seq, s, size_t(shift) * per_seq);
        }
      } else if (i < 20) {
        std::memset(dst, 0, m_gulp_bytes);
      }
      auto& mref = pld.get_mbuf()->GetMetadataRef();
      const uint64_t seq_start = m_opts.seq0 + uint64_t(i) * m_opts.nseq;
      mref["seq_start"] = seq_start;
      mref["seq_end"] = seq_start + m_opts.nseq;
      mref["nseqs"] = m_opts.nseq;
      mref["gulp_len_ms"] = m_opts.nseq * SAMPLING_LEN_uS * 1e-3;
      mref["nchan"] = uint8_t(m_opts.nchan);
      mref["chan0"] = int64_t(m_opts.chan0);
      mref["data_order"] = std::string("c_maj");
      mref["nbytes"] = int(m_gulp_bytes);
      mref["time_tag"] = seq_start * SEQ_MULT_S;
      mref["data_quality"] = 1.0;
      if (m_opts.realtime) {
        std::this_thread::sleep_until(t_start + std::chrono::duration_cast<clock::duration>(
                                                    std::chrono::duration<double>(i * gulp_s)));
      }
      auto now = clock::now();
      if (i > 0) interval_ms.push_back(std::chrono::duration<double, std::milli>(now - t_last).count());
      t_last = now;
      output["gulp"].push(pld);
      if ((i + 1) % 240 == 0) {
        double m, p50, p99, mx, wm, w50, w99, wmx;
        bench_detail::Summarise(interval_ms, &m, &p50, &p99, &mx);
        bench_detail::Summarise(wait_ms, &wm, &w50, &w99, &wmx);
        LOG(INFO) << "bench gen: " << (i + 1) << " gulps; interval mean " << m << " p50 " << p50 << " p99 " << p99
                  << " max " << mx << " ms (gulp spans " << gulp_s * 1e3 << " ms); waited for a buffer mean " << wm
                  << " p99 " << w99 << " ms";
        interval_ms.clear();
        wait_ms.clear();
      }
    }
    LOG(INFO) << "bench gen: done";
    return raft::stop;
  }

 private:
  BenchOptions m_opts;
  size_t m_gulp_bytes{0};
  std::unique_ptr<BufferMngr> m_buf_mngr;
  const uint8_t* m_data{nullptr};
  size_t m_map_bytes{0};
  int64_t m_nfile{0};
};

template <class Payload>
class BenchSink : public raft::kernel {
 public:
  BenchSink(std::string p_name, BenchOptions p_opts, bool p_dump)
      : raft::kernel(), m_name(std::move(p_name)), m_opts(std::move(p_opts)), m_dump(p_dump) {
    input.addPort<Payload>("in_img");
  }

  raft::kstatus run() override {
    Payload pld;
    input["in_img"].pop(pld);
    if (!pld) return raft::proceed;
    auto now = std::chrono::steady_clock::now();
    if (m_n > 0) m_iv.push_back(std::chrono::duration<double, std::milli>(now - m_last).count());
    m_last = now;
    ++m_n;
    auto& meta = pld.get_mbuf()->GetMetadataRef();
    uint64_t gv = 0, seq = 0;
    if (auto it = meta.find("gain_version"); it != meta.end()) gv = std::get<uint64_t>(it->second);
    if (auto it = meta.find("seq_start"); it != meta.end()) seq = std::get<uint64_t>(it->second);
    if (m_dump && !m_opts.dump_dir.empty() && m_opts.dump_every > 0 && (m_n % m_opts.dump_every) == 0) {
      int grid = std::get<int>(meta["grid_size"]);
      int nchan = std::get<uint8_t>(meta["nchan"]);
      size_t nbytes = size_t(nchan) * grid * grid * 4 * sizeof(float);
      char fn[512];
      std::snprintf(fn, sizeof(fn), "%s/img_seq%llu_gv%llu.f32", m_opts.dump_dir.c_str(), (unsigned long long)seq,
                    (unsigned long long)gv);
      std::ofstream(fn, std::ios::binary).write(reinterpret_cast<const char*>(pld.get_mbuf()->GetDataPtr()), nbytes);
    }
    if (m_opts.report_every > 0 && m_n % m_opts.report_every == 0) {
      double m, p50, p99, mx;
      bench_detail::Summarise(m_iv, &m, &p50, &p99, &mx);
      LOG(INFO) << "bench " << m_name << ": " << m_n << " images; interval mean " << m << " p50 " << p50 << " p99 "
                << p99 << " max " << mx << " ms; gain_version " << gv;
      m_iv.clear();
    }
    return raft::proceed;
  }

 private:
  std::string m_name;
  BenchOptions m_opts;
  bool m_dump;
  int64_t m_n{0};
  std::chrono::steady_clock::time_point m_last;
  std::vector<double> m_iv;
};

#endif  // SRC_RAFT_KERNELS_BENCH_KERNELS_HPP_
