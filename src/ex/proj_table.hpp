// Copyright (c) 2026 The EPIC++ authors
//
// Interference projector published by the CPU calibrator (streamcal) through
// a file in shared memory (/dev/shm), read by the imager at image boundaries.
// Every voltage sequence x (512 inputs, input = 2 stand + pol) of a channel is
// replaced by x - Ut (Vt^H x) before imaging (the oblique projection of
// docs/streaming_calibration/10_real_data_methodology.md, section 10.8, in the
// voltage domain). The calibration tap keeps the raw voltages.
//
// Layout (little endian), see src/python/streamcal/shm.py (ProjTableWriter):
//   ProjTableHeader (128 bytes)
//   std::complex<float> Ut[nchan][qmax][ninput]
//   std::complex<float> Vt[nchan][qmax][ninput]
//   std::complex<float> Sun[nchan][ninput]   (when flags bit 1 is set)
// Each component acts on the inputs of one polarisation; unused components
// are zero. Sun is the Sun's response on the raw inputs after projection, unit
// norm over each polarisation's inputs: the impulse detector removes the
// Sun-beam power |Sun^H x|^2 before its test, so solar bursts are not zeroed.
// Writers use a seqlock: `seq` is odd while the table is written.

#ifndef SRC_EX_PROJ_TABLE_HPP_
#define SRC_EX_PROJ_TABLE_HPP_

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "./constants.h"
#include "./rfi_stage.h"

struct ProjTableHeader {
  char magic[8];            // "EPICPROJ"
  uint32_t layout;          // 1
  uint32_t header_bytes;    // sizeof(ProjTableHeader)
  uint64_t seq;             // seqlock counter, even when stable
  uint64_t version;         // increases with every published table
  int64_t chan0;            // first channel (F-engine channel number)
  uint32_t nchan;           // channels in the table
  uint32_t ninput;          // LWA_SV_NSTANDS * LWA_SV_NPOLS
  uint32_t qmax;            // components stored per channel
  uint32_t q;               // components in use (the first q; 0: identity)
  uint32_t flags;           // bit 0: table valid (0 = projection off); bit 1: Sun vector present
  uint32_t reserved0;
  uint64_t valid_from_seq;  // apply only to gulps starting at or after this F-engine seq
  double t_solved_unix;     // time of the data the subspace is based on
  char solver[48];
};
static_assert(sizeof(ProjTableHeader) == 128, "ProjTableHeader must be 128 bytes");

struct ProjSnapshot {
  uint64_t version{0};
  int64_t chan0{0};
  uint32_t nchan{0};
  int q{0};  // 0: no projection
  bool valid{false};
  /// [nchan][kProjQmax][ninput], components beyond q are zero
  std::vector<std::complex<float>> ut, vt;
  /// [nchan][ninput] Sun response after projection (empty: none)
  std::vector<std::complex<float>> sun;
};

/**
 * Polls a projector table, like GainTableReader. Rejects tables whose vectors
 * are not a projector pair: in each polarisation Vt^H Ut must be the identity
 * on the components in use (to 5%).
 */
class ProjTableReader {
 public:
  explicit ProjTableReader(std::string p_path) : m_path(std::move(p_path)) {}
  ~ProjTableReader() { Unmap(); }
  ProjTableReader(const ProjTableReader&) = delete;
  ProjTableReader& operator=(const ProjTableReader&) = delete;

  std::optional<ProjSnapshot> Poll(int64_t p_chan0, uint32_t p_nchan, uint64_t p_seq) {
    constexpr uint32_t kIn = LWA_SV_NSTANDS * LWA_SV_NPOLS;
    if (!m_base && !Map()) {
      return std::nullopt;
    }
    auto* hdr = reinterpret_cast<const ProjTableHeader*>(m_base);
    const auto* seq_ptr = reinterpret_cast<const std::atomic<uint64_t>*>(&hdr->seq);
    uint64_t s1 = seq_ptr->load(std::memory_order_acquire);
    if (s1 & 1) {
      return std::nullopt;
    }
    ProjTableHeader h;
    std::memcpy(&h, hdr, sizeof(h));
    if (std::memcmp(h.magic, "EPICPROJ", 8) != 0 || h.layout != 1 || h.header_bytes != sizeof(ProjTableHeader) ||
        h.ninput != kIn || h.q > h.qmax) {
      m_last_error = "bad header";
      return std::nullopt;
    }
    if (h.version <= m_last_version || h.valid_from_seq > p_seq) {
      return std::nullopt;
    }
    const size_t per = size_t(h.qmax) * kIn;
    const bool has_sun = (h.flags & 2u) != 0;
    size_t need = sizeof(ProjTableHeader) + (2 * size_t(h.nchan) * per + (has_sun ? size_t(h.nchan) * kIn : 0)) *
                                                sizeof(std::complex<float>);
    if (need > m_size) {
      Unmap();
      if (!Map() || need > m_size) {
        return std::nullopt;
      }
      hdr = reinterpret_cast<const ProjTableHeader*>(m_base);
      seq_ptr = reinterpret_cast<const std::atomic<uint64_t>*>(&hdr->seq);
    }
    ProjSnapshot snap;
    snap.version = h.version;
    snap.valid = (h.flags & 1u) != 0 && (h.q > 0 || has_sun);
    snap.chan0 = p_chan0;
    snap.nchan = p_nchan;
    snap.q = snap.valid ? std::min<int>(int(h.q), kProjQmax) : 0;
    if (snap.valid) {
      if (p_chan0 < h.chan0 || p_chan0 + int64_t(p_nchan) > h.chan0 + int64_t(h.nchan)) {
        m_last_error = "table does not cover the imaged channels";
        return std::nullopt;
      }
      const auto* ut = reinterpret_cast<const std::complex<float>*>(m_base + sizeof(ProjTableHeader));
      const auto* vt = ut + size_t(h.nchan) * per;
      snap.ut.assign(size_t(p_nchan) * kProjQmax * kIn, {0.f, 0.f});
      snap.vt.assign(snap.ut.size(), {0.f, 0.f});
      for (uint32_t c = 0; c < p_nchan; ++c) {
        const size_t src = (size_t(p_chan0 - h.chan0) + c) * per;
        const size_t dst = size_t(c) * kProjQmax * kIn;
        std::memcpy(&snap.ut[dst], ut + src, size_t(snap.q) * kIn * sizeof(std::complex<float>));
        std::memcpy(&snap.vt[dst], vt + src, size_t(snap.q) * kIn * sizeof(std::complex<float>));
      }
      if (has_sun) {
        const auto* sun = vt + size_t(h.nchan) * per;
        snap.sun.assign(sun + size_t(p_chan0 - h.chan0) * kIn, sun + (size_t(p_chan0 - h.chan0) + p_nchan) * kIn);
      }
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    if (seq_ptr->load(std::memory_order_relaxed) != s1) {
      return std::nullopt;  // torn read, retry at the next poll
    }
    m_last_version = h.version;
    if (snap.valid && !Check(snap)) {
      return std::nullopt;  // m_last_error set; this version is not retried
    }
    m_last_error.clear();
    return snap;
  }

  void Reset() { m_last_version = 0; }
  const std::string& LastError() const { return m_last_error; }

 private:
  bool Check(const ProjSnapshot& p_snap) {
    constexpr int kIn = LWA_SV_NSTANDS * LWA_SV_NPOLS;
    for (size_t i = 0; i < p_snap.ut.size(); ++i) {
      const auto a = p_snap.ut[i], b = p_snap.vt[i];
      if (!std::isfinite(a.real()) || !std::isfinite(a.imag()) || !std::isfinite(b.real()) ||
          !std::isfinite(b.imag())) {
        m_last_error = "non-finite projector";
        return false;
      }
    }
    for (size_t i = 0; i < p_snap.sun.size(); ++i) {
      if (!std::isfinite(p_snap.sun[i].real()) || !std::isfinite(p_snap.sun[i].imag())) {
        m_last_error = "non-finite Sun vector";
        return false;
      }
    }
    for (uint32_t c = 0; c < p_snap.nchan; ++c) {
      const auto* ut = &p_snap.ut[size_t(c) * kProjQmax * kIn];
      const auto* vt = &p_snap.vt[size_t(c) * kProjQmax * kIn];
      for (int pol = 0; pol < LWA_SV_NPOLS; ++pol) {
        for (int k = 0; k < p_snap.q; ++k) {
          for (int l = 0; l < p_snap.q; ++l) {
            std::complex<double> g{0, 0};
            double nu = 0;
            for (int s = 0; s < LWA_SV_NSTANDS; ++s) {
              const int i = 2 * s + pol;
              g += std::conj(std::complex<double>(vt[k * kIn + i])) * std::complex<double>(ut[l * kIn + i]);
              nu += std::norm(std::complex<double>(ut[l * kIn + i]));
            }
            if (nu == 0) continue;  // unused component in this polarisation
            const double want = k == l ? 1.0 : 0.0;
            if (std::abs(g - want) > 0.05) {
              m_last_error = "not a projector pair (channel " + std::to_string(c) + ", pol " + std::to_string(pol) +
                             ", Vt^H Ut[" + std::to_string(k) + "][" + std::to_string(l) +
                             "] = " + std::to_string(std::abs(g)) + ")";
              return false;
            }
          }
        }
      }
    }
    return true;
  }
  bool Map() {
    int fd = ::open(m_path.c_str(), O_RDONLY);
    if (fd < 0) {
      return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || size_t(st.st_size) < sizeof(ProjTableHeader)) {
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
  const uint8_t* m_base{nullptr};
  size_t m_size{0};
  uint64_t m_last_version{0};
  std::string m_last_error;
};

#endif  // SRC_EX_PROJ_TABLE_HPP_
