// Copyright (c) 2026 The EPIC++ authors
//
// Visibility table shared with the CPU calibrator (streamcal): one file in
// /dev/shm holding
//   VisTableHeader (128 bytes)
//   std::complex<float> R[nchan][npol][nstand][nstand]   (R = sum / nsamp, Hermitian)
// under a seqlock (seq odd while writing). Written by the CPU calibration tap
// (cal_tap.hpp) or the GPU tap (tcc_tap.h).
#ifndef SRC_EX_VIS_TABLE_HPP_
#define SRC_EX_VIS_TABLE_HPP_

#include <fcntl.h>
#include <glog/logging.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>

struct VisTableHeader {
  char magic[8];          // "EPICVIS1"
  uint32_t layout;        // 1
  uint32_t header_bytes;  // sizeof(VisTableHeader)
  uint64_t seq;           // seqlock counter, even when stable
  uint64_t version;       // increases with every published table
  int64_t chan0;          // F-engine channel number of the first channel
  uint32_t nchan;
  uint32_t nstand;
  uint32_t npol;
  uint32_t flags;         // reserved
  uint64_t nsamp;         // samples integrated per channel
  uint64_t seq_first;     // F-engine sequence of the first tapped gulp
  uint64_t seq_last;      // F-engine sequence of the last tapped gulp (start)
  uint32_t ngulps;        // gulps correlated
  uint32_t nskipped;      // gulps offered but skipped (busy or low quality)
  uint64_t gulp_nseq;     // sequences per gulp
  float duty;             // fraction of each gulp's samples correlated
  char note[28];
};
static_assert(sizeof(VisTableHeader) == 128, "VisTableHeader must be 128 bytes");

class VisTableWriter {
 public:
  explicit VisTableWriter(std::string p_path) : m_path(std::move(p_path)) {}
  ~VisTableWriter() {
    if (m_map) munmap(m_map, m_map_size);
  }
  VisTableWriter(const VisTableWriter&) = delete;
  VisTableWriter& operator=(const VisTableWriter&) = delete;

  /// Publish one table. p_hdr supplies every field except magic, layout,
  /// header_bytes, seq and version; p_fill(R) writes the
  /// [nchan][npol][nstand][nstand] matrices. Returns the version, 0 on failure.
  template <class Fill>
  uint64_t Publish(const VisTableHeader& p_hdr, Fill&& p_fill) {
    const size_t bytes = sizeof(VisTableHeader) + size_t(p_hdr.nchan) * p_hdr.npol * p_hdr.nstand * p_hdr.nstand *
                                                      sizeof(std::complex<float>);
    if (!EnsureMapped(bytes)) {
      return 0;
    }
    auto* h = reinterpret_cast<VisTableHeader*>(m_map);
    auto* seq = reinterpret_cast<std::atomic<uint64_t>*>(&h->seq);
    seq->store(++m_seq, std::memory_order_relaxed);  // odd: writing
    std::atomic_thread_fence(std::memory_order_release);
    p_fill(reinterpret_cast<std::complex<float>*>(m_map + sizeof(VisTableHeader)));
    // every field but the seqlock counter
    constexpr size_t kSeqEnd = offsetof(VisTableHeader, seq) + sizeof(uint64_t);
    std::memcpy(m_map + kSeqEnd, reinterpret_cast<const uint8_t*>(&p_hdr) + kSeqEnd,
                sizeof(VisTableHeader) - kSeqEnd);
    std::memcpy(h->magic, "EPICVIS1", 8);
    h->layout = 1;
    h->header_bytes = sizeof(VisTableHeader);
    h->version = ++m_version;
    std::atomic_thread_fence(std::memory_order_release);
    seq->store(++m_seq, std::memory_order_release);  // even: stable
    return m_version;
  }

  uint64_t Version() const { return m_version; }

 private:
  bool EnsureMapped(size_t p_bytes) {
    if (m_map && m_map_size == p_bytes) {
      return true;
    }
    if (m_map) {
      munmap(m_map, m_map_size);
      m_map = nullptr;
    }
    int fd = ::open(m_path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0 || ftruncate(fd, p_bytes) != 0) {
      LOG(ERROR) << "VisTableWriter: cannot create " << m_path;
      if (fd >= 0) ::close(fd);
      return false;
    }
    void* p = mmap(nullptr, p_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) {
      LOG(ERROR) << "VisTableWriter: mmap failed for " << m_path;
      return false;
    }
    m_map = static_cast<uint8_t*>(p);
    m_map_size = p_bytes;
    auto* h = reinterpret_cast<VisTableHeader*>(m_map);
    m_seq = h->seq + (h->seq & 1);  // continue an existing counter
    return true;
  }

  std::string m_path;
  uint8_t* m_map{nullptr};
  size_t m_map_size{0};
  uint64_t m_seq{0};
  std::atomic<uint64_t> m_version{0};
};

#endif  // SRC_EX_VIS_TABLE_HPP_
