// Copyright (c) 2026 The EPIC++ authors
//
// Calibration tap: correlates a slice of every gulp on the CPU and publishes
// the integrated visibility matrices for the CPU calibrator (streamcal).
//
// The correlator thread offers each gulp with Offer(), which never blocks: the
// gulp pointer goes into a one-slot mailbox and the gulp is skipped if the
// stager is still busy. The stager thread copies the first `nseq` sequences
// of every channel into a staging batch (~3 MB per gulp for nseq = 50) while
// the correlator thread is blocked in ProcessGulp, and the correlator calls
// WaitCopied() before it releases the gulp. No Payload copy is shared between
// threads (Payload unlocks by checking use_count(), which is not thread safe).
//
// When a batch holds `batch` gulps, or the integration is complete, it is
// handed to the X-engine thread (xengine.hpp) and the stager continues in the
// other of two staging buffers. Correlating several gulps per call amortises
// the read-modify-write of the 134 MB accumulator, which dominates at
// nseq = 50: with one gulp per call the tap kept up with every second gulp.
// After `interval_s` of F-engine time the X-engine thread publishes the
// visibility table (vis_table.hpp). The voltages
// are the raw F-engine samples, before any phases or gains, so the solution
// does not depend on the gains the imager applied.
#ifndef SRC_EX_CAL_TAP_HPP_
#define SRC_EX_CAL_TAP_HPP_

#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <complex>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "./constants.h"
#include "./vis_table.hpp"
#include "./xengine.hpp"

struct CalTapOptions {
  std::string path;          // output file, e.g. /dev/shm/epic_vis_gpu0
  int nseq{0};               // sequences correlated per gulp (0 = tap off)
  float interval_s{10.f};    // integration per published table
  int nthreads{2};           // X-engine threads
  int batch{8};              // gulps correlated per X-engine call
  float min_quality{0.98f};  // skip gulps with less valid data than this
};

class CalTap {
 public:
  explicit CalTap(CalTapOptions p_opts) : m_opts(std::move(p_opts)), m_writer(m_opts.path) {
    m_opts.batch = std::max(1, m_opts.batch);
    m_stager = std::thread([this] { StageLoop(); });
    m_xeng = std::thread([this] { XengLoop(); });
  }
  ~CalTap() {
    {
      std::lock_guard<std::mutex> lk(m_mtx);
      m_stop = true;
    }
    m_cv.notify_all();
    m_cv_copied.notify_all();
    m_cv_batch.notify_all();
    if (m_stager.joinable()) m_stager.join();
    if (m_xeng.joinable()) m_xeng.join();
  }
  CalTap(const CalTap&) = delete;
  CalTap& operator=(const CalTap&) = delete;

  /// Offer a gulp. Returns false if it was skipped. Never blocks for long.
  /// If it returns true the caller must call WaitCopied() before the gulp
  /// buffer is released or reused.
  bool Offer(const uint8_t* p_data, uint64_t p_seq_start, int64_t p_chan0, int p_nchan, int p_gulp_nseq,
             float p_quality) {
    std::unique_lock<std::mutex> lk(m_mtx, std::try_to_lock);
    if (!lk.owns_lock() || m_slot || m_staging || p_quality < m_opts.min_quality || !p_data) {
      m_nskipped.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    m_slot = Item{p_data, p_seq_start, p_chan0, p_nchan, p_gulp_nseq};
    m_copied = false;
    m_pending = true;
    lk.unlock();
    m_cv.notify_one();
    return true;
  }

  /// Block until the stager has copied the last offered gulp (normally already
  /// done: staging takes ~0.5 ms, ProcessGulp ~35 ms).
  void WaitCopied() {
    std::unique_lock<std::mutex> lk(m_mtx);
    if (!m_pending) return;
    m_cv_copied.wait(lk, [this] { return m_copied || m_stop; });
    m_pending = false;
  }

  uint64_t Version() const { return m_writer.Version(); }

 private:
  struct Item {
    const uint8_t* data;  // valid until WaitCopied() returns
    uint64_t seq_start;
    int64_t chan0;
    int nchan;
    int gulp_nseq;
  };
  struct Batch {
    std::vector<uint8_t> data;  // [chan][batch * nseq][512]
    int nfill{0};
    int nseq{0};                // sequences per gulp slice
    int64_t chan0{-1};
    int nchan{0};
    int gulp_nseq{0};
    uint64_t seq_first{0}, seq_last{0};
    bool publish{false};        // last batch of the integration
  };

  // ---- stager thread
  void StageLoop() {
    while (true) {
      std::optional<Item> item;
      {
        std::unique_lock<std::mutex> lk(m_mtx);
        m_cv.wait(lk, [this] { return m_stop || m_slot.has_value(); });
        if (m_stop) return;
        item.swap(m_slot);
        m_staging = true;
      }
      Stage(*item);
      {
        std::lock_guard<std::mutex> lk(m_mtx);
        m_copied = true;
        m_staging = false;
      }
      m_cv_copied.notify_all();
    }
  }

  void Stage(const Item& p_it) {
    Batch& b = m_batch[m_cur];
    const int nseq = std::min(m_opts.nseq, p_it.gulp_nseq);
    if (b.nfill > 0 && (b.chan0 != p_it.chan0 || b.nchan != p_it.nchan || b.nseq != nseq)) {
      b.nfill = 0;  // spectral window changed: drop the partial batch
      m_int_open = false;
    }
    if (b.nfill == 0) {
      b.chan0 = p_it.chan0;
      b.nchan = p_it.nchan;
      b.nseq = nseq;
      b.gulp_nseq = p_it.gulp_nseq;
      b.seq_first = p_it.seq_start;
      b.data.resize(size_t(p_it.nchan) * m_opts.batch * nseq * LWA_SV_INP_PER_CHAN);
    }
    if (!m_int_open || m_int_chan0 != p_it.chan0) {
      m_int_first = p_it.seq_start;
      m_int_chan0 = p_it.chan0;
      m_int_open = true;
    }
    const size_t slice = size_t(nseq) * LWA_SV_INP_PER_CHAN;
    const size_t chan_stride = size_t(m_opts.batch) * slice;
    for (int c = 0; c < p_it.nchan; ++c) {
      std::memcpy(b.data.data() + c * chan_stride + b.nfill * slice,
                  p_it.data + size_t(c) * p_it.gulp_nseq * LWA_SV_INP_PER_CHAN, slice);
    }
    ++b.nfill;
    b.seq_last = p_it.seq_start;
    const double span_s = double(p_it.seq_start + p_it.gulp_nseq - m_int_first) * SAMPLING_LEN_uS * 1e-6;
    b.publish = span_s >= m_opts.interval_s;
    if (b.nfill == m_opts.batch || b.publish) {
      std::unique_lock<std::mutex> lk(m_mtx);
      if (m_ready) {
        // the X-engine has not finished the previous batch: drop this one
        m_nskipped.fetch_add(b.nfill, std::memory_order_relaxed);
        b.nfill = 0;
        return;
      }
      m_ready = &b;
      m_cur ^= 1;
      if (b.publish) m_int_open = false;
      lk.unlock();
      m_cv_batch.notify_one();
    }
  }

  // ---- X-engine thread
  void XengLoop() {
    while (true) {
      Batch* b = nullptr;
      {
        std::unique_lock<std::mutex> lk(m_mtx);
        m_cv_batch.wait(lk, [this] { return m_stop || m_ready != nullptr; });
        if (m_stop) return;
        b = m_ready;
      }
      if (b->chan0 != m_chan0 || b->nchan != m_nchan || b->gulp_nseq != m_gulp_nseq) {
        Reset(b->chan0, b->nchan, b->gulp_nseq);
      }
      if (m_ngulps == 0) m_seq_first = b->seq_first;
      // each channel holds batch * nseq staged sequences, of which nfill * nseq are filled
      const int stride_seq = m_opts.batch * b->nseq;
      const int nseq_total = b->nfill * b->nseq;
      xengine::accumulate(b->data.data(), stride_seq, 0, b->nchan, 0, nseq_total, m_re.data(), m_im.data(),
                          m_opts.nthreads);
      m_nsamp += nseq_total;
      m_ngulps += b->nfill;
      m_seq_last = b->seq_last;
      const bool publish = b->publish;
      {
        std::lock_guard<std::mutex> lk(m_mtx);
        b->nfill = 0;
        m_ready = nullptr;
      }
      if (publish) {
        Publish();
        Reset(m_chan0, m_nchan, m_gulp_nseq);
      }
    }
  }

  void Reset(int64_t p_chan0, int p_nchan, int p_gulp_nseq) {
    m_chan0 = p_chan0;
    m_nchan = p_nchan;
    m_gulp_nseq = p_gulp_nseq;
    const size_t n = size_t(p_nchan) * LWA_SV_NPOLS * LWA_SV_NSTANDS * LWA_SV_NSTANDS;
    m_re.assign(n, 0);
    m_im.assign(n, 0);
    m_nsamp = 0;
    m_ngulps = 0;
    m_seq_first = 0;
  }

  void Publish() {
    if (m_nsamp == 0) {
      return;
    }
    constexpr size_t kN = LWA_SV_NSTANDS;
    VisTableHeader h{};
    h.chan0 = m_chan0;
    h.nchan = m_nchan;
    h.nstand = kN;
    h.npol = LWA_SV_NPOLS;
    h.nsamp = m_nsamp;
    h.seq_first = m_seq_first;
    h.seq_last = m_seq_last;
    h.ngulps = m_ngulps;
    h.nskipped = uint32_t(m_nskipped.exchange(0));
    h.gulp_nseq = m_gulp_nseq;
    h.duty = float(std::min(m_opts.nseq, m_gulp_nseq)) / float(m_gulp_nseq);
    std::strncpy(h.note, "xengine-avx2 v1", sizeof(h.note) - 1);
    const float inv = 1.f / float(m_nsamp);
    const uint64_t v = m_writer.Publish(h, [&](std::complex<float>* R) {
      for (size_t cp = 0; cp < size_t(m_nchan) * LWA_SV_NPOLS; ++cp) {
        const int64_t* re = m_re.data() + cp * kN * kN;
        const int64_t* im = m_im.data() + cp * kN * kN;
        std::complex<float>* out = R + cp * kN * kN;
        for (size_t a = 0; a < kN; ++a) {
          for (size_t b = a; b < kN; ++b) {
            std::complex<float> z{float(re[a * kN + b]) * inv, float(im[a * kN + b]) * inv};
            out[a * kN + b] = z;
            out[b * kN + a] = std::conj(z);
          }
        }
      }
    });
    VLOG(1) << "CalTap: published version " << v << " (" << m_ngulps << " gulps, " << m_nsamp
            << " samples, chan0 " << m_chan0 << ")";
  }

  CalTapOptions m_opts;
  std::thread m_stager, m_xeng;
  std::mutex m_mtx;
  std::condition_variable m_cv, m_cv_copied, m_cv_batch;
  std::optional<Item> m_slot;
  bool m_staging{false};
  bool m_copied{false};
  bool m_pending{false};
  bool m_stop{false};
  std::atomic<uint64_t> m_nskipped{0};

  // staging (stager thread; a full batch is handed over through m_ready)
  Batch m_batch[2];
  int m_cur{0};
  Batch* m_ready{nullptr};
  uint64_t m_int_first{0};
  bool m_int_open{false};  // an integration has started
  int64_t m_int_chan0{-1};

  // accumulation (X-engine thread)
  int64_t m_chan0{-1};
  int m_nchan{0};
  int m_gulp_nseq{0};
  std::vector<int64_t> m_re, m_im;
  uint64_t m_nsamp{0};
  uint32_t m_ngulps{0};
  uint64_t m_seq_first{0}, m_seq_last{0};

  VisTableWriter m_writer;
};

#endif  // SRC_EX_CAL_TAP_HPP_
