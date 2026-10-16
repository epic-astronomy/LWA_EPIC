// Copyright (c) 2026 The EPIC++ authors
//
// Per-antenna calibration gains published by the CPU calibrator (streamcal)
// through a file in shared memory (/dev/shm), read by the imager between
// gulps and multiplied into the nominal per-antenna phases (phases_g).
//
// Layout (little endian), see src/python/streamcal/gaintable.py:
//   GainTableHeader (128 bytes)
//   std::complex<float> corr[nchan][LWA_SV_NSTANDS][LWA_SV_NPOLS]
// corr multiplies the nominal phases (geometry x cable model) as is: it is
// 1/g for the residual gain g the calibrator solved for. A zero entry flags the
// stand in that channel. Writers use a seqlock: `seq` is odd while the table
// is being written and is incremented again when it is complete.

#ifndef SRC_EX_GAIN_TABLE_HPP_
#define SRC_EX_GAIN_TABLE_HPP_

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "./constants.h"

struct GainTableHeader {
  char magic[8];             // "EPICGAIN"
  uint32_t layout;           // 1
  uint32_t header_bytes;     // sizeof(GainTableHeader)
  uint64_t seq;              // seqlock counter, even when stable
  uint64_t version;          // increases with every published table
  int64_t chan0;             // first channel (F-engine channel number)
  uint32_t nchan;            // channels in the table
  uint32_t nstand;           // LWA_SV_NSTANDS
  uint32_t npol;             // LWA_SV_NPOLS
  uint32_t flags;            // bit 0: table valid (0 = revert to nominal)
  uint64_t valid_from_seq;   // apply only to gulps starting at or after this F-engine seq
  double t_solved_unix;      // time of the data the solution is based on
  float max_abs;             // writer-side clamp, for information
  uint32_t reserved0;
  char solver[48];           // free text, e.g. "stefcal+kalman v1"
};
static_assert(sizeof(GainTableHeader) == 128, "GainTableHeader must be 128 bytes");

struct GainSnapshot {
  uint64_t version{0};
  int64_t chan0{0};
  uint32_t nchan{0};
  bool valid{false};
  std::vector<std::complex<float>> corr;  // [nchan][stand][pol]
};

/**
 * Polls a gain table file. Cheap when nothing changed (one 8-byte read). Safe
 * against a writer that crashes mid-update (the seqlock stays odd and the
 * table is ignored) and against a missing file (retried on every poll).
 */
class GainTableReader {
 public:
  explicit GainTableReader(std::string p_path, float p_max_abs = 64.f)
      : m_path(std::move(p_path)), m_max_abs(p_max_abs) {}
  ~GainTableReader() { Unmap(); }
  GainTableReader(const GainTableReader&) = delete;
  GainTableReader& operator=(const GainTableReader&) = delete;

  /// Returns a new table when one with a higher version than the last returned
  /// is available, covers [p_chan0, p_chan0 + p_nchan) and is valid for p_seq.
  std::optional<GainSnapshot> Poll(int64_t p_chan0, uint32_t p_nchan, uint64_t p_seq) {
    if (!m_base && !Map()) {
      return std::nullopt;
    }
    auto* hdr = reinterpret_cast<const GainTableHeader*>(m_base);
    const auto* seq_ptr = reinterpret_cast<const std::atomic<uint64_t>*>(&hdr->seq);
    uint64_t s1 = seq_ptr->load(std::memory_order_acquire);
    if (s1 & 1) {
      return std::nullopt;  // being written
    }
    GainTableHeader h;
    std::memcpy(&h, hdr, sizeof(h));
    if (std::memcmp(h.magic, "EPICGAIN", 8) != 0 || h.layout != 1 ||
        h.header_bytes != sizeof(GainTableHeader) || h.nstand != LWA_SV_NSTANDS ||
        h.npol != LWA_SV_NPOLS) {
      m_last_error = "bad header";
      return std::nullopt;
    }
    if (h.version <= m_last_version || h.valid_from_seq > p_seq) {
      return std::nullopt;
    }
    size_t need = sizeof(GainTableHeader) + size_t(h.nchan) * h.nstand * h.npol * sizeof(std::complex<float>);
    if (need > m_size) {
      // the writer grew the file: remap
      Unmap();
      if (!Map() || need > m_size) {
        return std::nullopt;
      }
      hdr = reinterpret_cast<const GainTableHeader*>(m_base);
      seq_ptr = reinterpret_cast<const std::atomic<uint64_t>*>(&hdr->seq);
    }
    GainSnapshot snap;
    snap.version = h.version;
    snap.valid = (h.flags & 1u) != 0;
    snap.chan0 = p_chan0;
    snap.nchan = p_nchan;
    if (snap.valid) {
      if (p_chan0 < h.chan0 || p_chan0 + int64_t(p_nchan) > h.chan0 + int64_t(h.nchan)) {
        m_last_error = "table does not cover the imaged channels";
        return std::nullopt;
      }
      const size_t per_chan = size_t(h.nstand) * h.npol;
      snap.corr.resize(size_t(p_nchan) * per_chan);
      const auto* src = reinterpret_cast<const std::complex<float>*>(m_base + sizeof(GainTableHeader)) +
                        size_t(p_chan0 - h.chan0) * per_chan;
      std::memcpy(snap.corr.data(), src, snap.corr.size() * sizeof(std::complex<float>));
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    if (seq_ptr->load(std::memory_order_relaxed) != s1) {
      return std::nullopt;  // torn read, retry at the next poll
    }
    for (auto& c : snap.corr) {
      if (!std::isfinite(c.real()) || !std::isfinite(c.imag()) || std::abs(c) > m_max_abs) {
        m_last_error = "non-finite or out-of-range gain";
        m_last_version = h.version;  // do not retry this table
        return std::nullopt;
      }
    }
    m_last_version = h.version;
    m_last_error.clear();
    return snap;
  }

  /// Forget the last version, e.g. after the spectral window changed.
  void Reset() { m_last_version = 0; }
  const std::string& LastError() const { return m_last_error; }

 private:
  bool Map() {
    int fd = ::open(m_path.c_str(), O_RDONLY);
    if (fd < 0) {
      return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || size_t(st.st_size) < sizeof(GainTableHeader)) {
      ::close(fd);
      return false;
    }
    void* p = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) {
      return false;
    }
    m_base = static_cast<const uint8_t*>(p);
    m_size = st.st_size;
    return true;
  }
  void Unmap() {
    if (m_base) {
      munmap(const_cast<uint8_t*>(m_base), m_size);
      m_base = nullptr;
      m_size = 0;
    }
  }

  std::string m_path;
  float m_max_abs;
  const uint8_t* m_base{nullptr};
  size_t m_size{0};
  uint64_t m_last_version{0};
  std::string m_last_error;
};

/// Multiply the nominal phases [chan][stand][pol][re,im] by the table.
inline void ApplyGainCorrections(const float* p_nominal, const GainSnapshot& p_snap, float* p_out) {
  const size_t n = size_t(p_snap.nchan) * LWA_SV_NSTANDS * LWA_SV_NPOLS;
  if (!p_snap.valid) {
    std::memcpy(p_out, p_nominal, n * 2 * sizeof(float));
    return;
  }
  for (size_t i = 0; i < n; ++i) {
    std::complex<float> nom(p_nominal[2 * i], p_nominal[2 * i + 1]);
    std::complex<float> v = nom * p_snap.corr[i];
    p_out[2 * i] = v.real();
    p_out[2 * i + 1] = v.imag();
  }
}

#endif  // SRC_EX_GAIN_TABLE_HPP_
