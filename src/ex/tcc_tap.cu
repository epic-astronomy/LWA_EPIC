// Copyright (c) 2026 The EPIC++ authors
//
// GPU calibration tap with the Tensor-Core Correlator; see tcc_tap.h.
#include "./tcc_tap.h"

#include <glog/logging.h>

#ifdef EPIC_WITH_TCC

#include <algorithm>
#include <complex>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "./constants.h"
#include "./cpu_reserve.hpp"
#include "./vis_table.hpp"
#include "libtcc/Correlator.h"

namespace {
constexpr int kStands = LWA_SV_NSTANDS;
constexpr int kPols = LWA_SV_NPOLS;
constexpr int kInputs = kStands * kPols;
constexpr int kT = 16;  // TCC i8 samples per block
constexpr size_t kNbl = size_t(kStands) * (kStands + 1) / 2;

void Check(cudaError_t p_err, const char* p_what) {
  LOG_IF(FATAL, p_err != cudaSuccess) << "TccTap: " << p_what << ": " << cudaGetErrorString(p_err);
}

// [chan][seq][input] CNib bytes (re high nibble, im low) -> TCC i8
// [chan][seq / 16][input][16][re, im]; sequences beyond nseq are zero.
// One block per (16-sequence block, channel), one thread per input.
__global__ void ReorderToTccI8(const uint8_t* __restrict__ p_in, int8_t* __restrict__ p_out, int p_nseq,
                               int p_ntb) {
  const int i = threadIdx.x;
  const int tb = blockIdx.x;
  const int c = blockIdx.y;
  const uint8_t* src = p_in + size_t(c) * p_nseq * kInputs + i;
  alignas(16) int8_t v[2 * kT];
#pragma unroll
  for (int tt = 0; tt < kT; ++tt) {
    const int t = tb * kT + tt;
    const uint8_t b = t < p_nseq ? src[size_t(t) * kInputs] : 0;
    v[2 * tt] = int8_t(b) >> 4;
    v[2 * tt + 1] = int8_t(uint8_t(b << 4)) >> 4;
  }
  uint4* dst = reinterpret_cast<uint4*>(p_out + ((size_t(c) * p_ntb + tb) * kInputs + i) * 2 * kT);
  dst[0] = reinterpret_cast<const uint4*>(v)[0];
  dst[1] = reinterpret_cast<const uint4*>(v)[1];
}

// Sample weights in the same layout: 1 if the sample's packet arrived (bit t
// of mask[(board * NDP_CHAN_BLOCKS + block[c]) * words]), else 0; the
// correlated weights count the sequences in which both samples of a product
// were valid (as FxImager does for the images)
__global__ void WeightsToTccI8(const uint64_t* __restrict__ p_mask, const uint8_t* __restrict__ p_block,
                               int8_t* __restrict__ p_out, int p_nseq, int p_ntb, int p_words) {
  const int i = threadIdx.x, tb = blockIdx.x, c = blockIdx.y;
  const uint64_t* m = p_mask + size_t((i / (kInputs / NROACH_BOARDS)) * NDP_CHAN_BLOCKS + p_block[c]) * p_words;
  alignas(16) int8_t v[2 * kT];
#pragma unroll
  for (int tt = 0; tt < kT; ++tt) {
    const int t = tb * kT + tt;
    v[2 * tt] = int8_t(t < p_nseq && ((m[t / 64] >> (t % 64)) & 1));
    v[2 * tt + 1] = 0;
  }
  uint4* dst = reinterpret_cast<uint4*>(p_out + ((size_t(c) * p_ntb + tb) * kInputs + i) * 2 * kT);
  dst[0] = reinterpret_cast<const uint4*>(v)[0];
  dst[1] = reinterpret_cast<const uint4*>(v)[1];
}
}  // namespace

struct TccTap::Impl {
  TccTapOptions opts;
  int device, nchan, nseq, nseq_pad, nstreams;
  size_t in_bytes, vis_bytes;

  std::unique_ptr<tcc::Correlator> corr;
  cudaStream_t s_tap{nullptr}, s_copy{nullptr};
  std::vector<cudaEvent_t> ev_copied;
  cudaEvent_t ev_read{nullptr}, ev_corr{nullptr}, ev_d2h[2]{nullptr, nullptr};
  int8_t* d_in{nullptr};
  void* d_vis[2]{nullptr, nullptr};
  int32_t* h_vis{nullptr};
  // sample weights: the next gulp's validity (SetGulpValidity), their TCC
  // input and correlation; weighted: some gulp of the integration lost packets
  std::vector<uint64_t> mask;
  std::vector<uint8_t> block;
  bool have_mask{false}, weighted{false}, pub_weighted{false};
  int mask_words{0};
  uint64_t* d_mask{nullptr};
  uint8_t* d_block{nullptr};
  int8_t* d_win{nullptr};
  void* d_wvis[2]{nullptr, nullptr};
  int32_t* h_wvis{nullptr};

  // integration (imaging thread)
  bool armed{false};
  int cur{0};
  bool open{false};  // the current buffer holds part of an integration
  int64_t chan0{-1};
  uint64_t seq_first{0}, seq_last{0}, nsamp{0}, nskipped{0};
  uint64_t weighted_base{0}, pub_weighted_base{0};  // complete samples before the first weighted gulp
  uint32_t ngulps{0};
  uint64_t armed_seq{0};

  // publisher thread
  std::thread pub;
  std::mutex mtx;
  std::condition_variable cv;
  bool busy{false}, stop{false};
  int pub_buf{0};
  VisTableHeader pub_hdr{};
  VisTableWriter writer;

  Impl(TccTapOptions p_opts, int p_device, int p_nchan, int p_nseq, int p_nstreams)
      : opts(std::move(p_opts)),
        device(p_device),
        nchan(p_nchan),
        nseq(p_nseq),
        nseq_pad((p_nseq + kT - 1) / kT * kT),
        nstreams(p_nstreams),
        writer(opts.path) {
    // 4-bit samples: |x|^2 <= 128, so the int32 accumulators hold
    // 2^31 / 128 samples, 700 s of F-engine data.
    LOG_IF(FATAL, opts.interval_s > 600.f) << "TccTap: integration must be at most 600 s (int32 accumulators)";
    Check(cudaSetDevice(device), "cudaSetDevice");
    Check(cudaFree(nullptr), "context");  // make the primary context current for libtcc
    cu::init();
    cu::Device dev(device);
    corr = std::make_unique<tcc::Correlator>(dev, tcc::i8, kStands, nchan, nseq_pad, kPols);
    in_bytes = size_t(nchan) * nseq_pad * kInputs * 2;
    vis_bytes = size_t(nchan) * kNbl * kPols * kPols * 2 * sizeof(int32_t);
    Check(cudaMalloc(&d_in, in_bytes), "cudaMalloc input");
    Check(cudaMalloc(&d_vis[0], vis_bytes), "cudaMalloc visibilities");
    Check(cudaMalloc(&d_vis[1], vis_bytes), "cudaMalloc visibilities");
    Check(cudaMallocHost(&h_vis, vis_bytes), "cudaMallocHost");
    mask_words = (nseq + 63) / 64;
    Check(cudaMalloc(&d_mask, size_t(NROACH_BOARDS) * NDP_CHAN_BLOCKS * mask_words * sizeof(uint64_t)), "mask");
    Check(cudaMalloc(&d_block, size_t(nchan)), "blocks");
    Check(cudaMalloc(&d_win, in_bytes), "cudaMalloc weights input");
    Check(cudaMalloc(&d_wvis[0], vis_bytes), "cudaMalloc weights");
    Check(cudaMalloc(&d_wvis[1], vis_bytes), "cudaMalloc weights");
    Check(cudaMallocHost(&h_wvis, vis_bytes), "cudaMallocHost");
    Check(cudaStreamCreateWithFlags(&s_tap, cudaStreamNonBlocking), "stream");
    Check(cudaStreamCreateWithFlags(&s_copy, cudaStreamNonBlocking), "stream");
    ev_copied.resize(nstreams);
    for (auto& e : ev_copied) Check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
    Check(cudaEventCreateWithFlags(&ev_read, cudaEventDisableTiming), "event");
    Check(cudaEventCreateWithFlags(&ev_corr, cudaEventDisableTiming), "event");
    Check(cudaEventCreateWithFlags(&ev_d2h[0], cudaEventDisableTiming), "event");
    Check(cudaEventCreateWithFlags(&ev_d2h[1], cudaEventDisableTiming), "event");
    pub = std::thread([this] { PublishLoop(); });
    LOG(INFO) << "TccTap: tensor-core correlator for " << nchan << " channels x " << nseq << " sequences ("
              << nseq_pad << " padded) per gulp; " << opts.interval_s << " s tables to " << opts.path << "; "
              << (in_bytes + 2 * vis_bytes) / (1 << 20) << " MiB on GPU " << device;
  }

  ~Impl() {
    {
      std::lock_guard<std::mutex> lk(mtx);
      stop = true;
    }
    cv.notify_all();
    if (pub.joinable()) pub.join();
    cudaSetDevice(device);
    cudaStreamSynchronize(s_tap);
    cudaStreamSynchronize(s_copy);
    for (auto& e : ev_copied) cudaEventDestroy(e);
    cudaEventDestroy(ev_read);
    cudaEventDestroy(ev_corr);
    cudaEventDestroy(ev_d2h[0]);
    cudaEventDestroy(ev_d2h[1]);
    cudaStreamDestroy(s_tap);
    cudaStreamDestroy(s_copy);
    cudaFree(d_in);
    cudaFree(d_vis[0]);
    cudaFree(d_vis[1]);
    cudaFreeHost(h_vis);
    cudaFree(d_mask);
    cudaFree(d_block);
    cudaFree(d_win);
    cudaFree(d_wvis[0]);
    cudaFree(d_wvis[1]);
    cudaFreeHost(h_wvis);
  }

  void SetGulpValidity(const uint64_t* p_mask, size_t p_words, const uint8_t* p_block) {
    have_mask = false;
    if (!p_mask || !p_block || int(p_words) != mask_words) return;
    const size_t n = size_t(NROACH_BOARDS) * NDP_CHAN_BLOCKS * p_words;
    const uint64_t last = nseq % 64 ? (uint64_t(1) << (nseq % 64)) - 1 : ~uint64_t(0);
    for (size_t i = 0; i < n && !have_mask; ++i) {
      have_mask = (p_mask[i] | ((i + 1) % p_words == 0 ? ~last : 0)) != ~uint64_t(0);  // a packet lost
    }
    if (!have_mask) return;
    mask.assign(p_mask, p_mask + n);
    block.assign(p_block, p_block + nchan);
  }

  bool Arm(uint64_t p_seq_start, int64_t p_chan0, int p_nchan, int p_nseq, float p_quality) {
    armed = false;
    if (p_nchan < nchan || p_nseq != nseq || p_quality < opts.min_quality) {
      ++nskipped;
      return false;
    }
    if (p_chan0 != chan0) {  // new spectral window: drop the partial integration
      chan0 = p_chan0;
      open = false;
    }
    armed = true;
    armed_seq = p_seq_start;
    return true;
  }

  void Launch(const uint8_t* p_gulp) {
    if (!armed) {
      return;
    }
    armed = false;
    for (auto& e : ev_copied) Check(cudaStreamWaitEvent(s_tap, e, 0), "wait copy");
    const int ntb = nseq_pad / kT;
    ReorderToTccI8<<<dim3(ntb, nchan), kInputs, 0, s_tap>>>(p_gulp, d_in, nseq, ntb);
    Check(cudaGetLastError(), "reorder launch");
    Check(cudaEventRecord(ev_read, s_tap), "record");
    if (!open) {
      // the buffer's previous table must have reached the host
      Check(cudaStreamWaitEvent(s_tap, ev_d2h[cur], 0), "wait d2h");
      seq_first = armed_seq;
      nsamp = 0;
      ngulps = 0;
      weighted = false;
      weighted_base = 0;
    }
    corr->launchAsync(reinterpret_cast<CUstream>(s_tap), reinterpret_cast<CUdeviceptr>(d_vis[cur]),
                      reinterpret_cast<CUdeviceptr>(d_in), open);
    // The weights are correlated from the first gulp of the integration that
    // lost packets on; the complete gulps before it add their samples on the host
    if (have_mask || weighted) {
      std::vector<uint64_t> full;
      if (!have_mask) {  // a complete gulp of a weighted integration
        full.assign(size_t(NROACH_BOARDS) * NDP_CHAN_BLOCKS * mask_words, ~uint64_t(0));
      }
      const uint64_t* hm = have_mask ? mask.data() : full.data();
      Check(cudaMemcpyAsync(d_mask, hm, size_t(NROACH_BOARDS) * NDP_CHAN_BLOCKS * mask_words * sizeof(uint64_t),
                            cudaMemcpyHostToDevice, s_tap), "mask");
      if (have_mask) Check(cudaMemcpyAsync(d_block, block.data(), nchan, cudaMemcpyHostToDevice, s_tap), "blocks");
      WeightsToTccI8<<<dim3(ntb, nchan), kInputs, 0, s_tap>>>(d_mask, d_block, d_win, nseq, ntb, mask_words);
      Check(cudaGetLastError(), "weights launch");
      if (!weighted) weighted_base = nsamp;  // the complete gulps before, all samples valid
      corr->launchAsync(reinterpret_cast<CUstream>(s_tap), reinterpret_cast<CUdeviceptr>(d_wvis[cur]),
                        reinterpret_cast<CUdeviceptr>(d_win), weighted);
      weighted = true;
    }
    have_mask = false;
    open = true;
    nsamp += nseq;
    ++ngulps;
    seq_last = armed_seq;
    const double span_s = double(armed_seq + nseq - seq_first) * SAMPLING_LEN_uS * 1e-6;
    if (span_s < opts.interval_s) {
      return;
    }
    std::unique_lock<std::mutex> lk(mtx);
    if (busy) {
      return;  // the previous table is still being converted: integrate longer
    }
    Check(cudaEventRecord(ev_corr, s_tap), "record");
    Check(cudaStreamWaitEvent(s_copy, ev_corr, 0), "wait corr");
    Check(cudaMemcpyAsync(h_vis, d_vis[cur], vis_bytes, cudaMemcpyDeviceToHost, s_copy), "d2h");
    if (weighted) Check(cudaMemcpyAsync(h_wvis, d_wvis[cur], vis_bytes, cudaMemcpyDeviceToHost, s_copy), "d2h");
    pub_weighted = weighted;
    pub_weighted_base = weighted_base;
    Check(cudaEventRecord(ev_d2h[cur], s_copy), "record");
    VisTableHeader& h = pub_hdr;
    h = VisTableHeader{};
    h.chan0 = chan0;
    h.nchan = nchan;
    h.nstand = kStands;
    h.npol = kPols;
    h.nsamp = nsamp;
    h.seq_first = seq_first;
    h.seq_last = seq_last;
    h.ngulps = ngulps;
    h.nskipped = uint32_t(nskipped);
    h.gulp_nseq = nseq;
    h.duty = 1.f;
    std::strncpy(h.note, "tcc-i8 v1", sizeof(h.note) - 1);
    nskipped = 0;
    pub_buf = cur;
    busy = true;
    lk.unlock();
    cv.notify_one();
    cur ^= 1;
    open = false;
  }

  void PublishLoop() {
    Check(cudaSetDevice(device), "cudaSetDevice");
    while (true) {
      // off the packet receiver's CPU (reserved after this thread started)
      cpus::UseFree();
      VisTableHeader h;
      int buf;
      bool wtd;
      uint64_t wbase;
      {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [this] { return stop || busy; });
        if (stop) return;
        h = pub_hdr;
        buf = pub_buf;
        wtd = pub_weighted;
        wbase = pub_weighted_base;
      }
      Check(cudaEventSynchronize(ev_d2h[buf]), "sync d2h");
      // TCC: v[chan][bl][polY][polX] = sum x[recvY, polY] conj(x[recvX, polX]),
      // bl = recvX (recvX + 1) / 2 + recvY, recvY <= recvX
      const float inv = 1.f / float(h.nsamp);
      const uint64_t v = writer.Publish(h, [&](std::complex<float>* R) {
        for (int c = 0; c < nchan; ++c) {
          for (int p = 0; p < kPols; ++p) {
            std::complex<float>* out = R + (size_t(c) * kPols + p) * kStands * kStands;
            for (int b = 0; b < kStands; ++b) {
              const int32_t* row = h_vis + ((size_t(c) * kNbl + size_t(b) * (b + 1) / 2) * kPols * kPols +
                                            (p * kPols + p)) * 2;
              const int32_t* wrow = h_wvis + (row - h_vis);
              for (int a = 0; a <= b; ++a) {
                const int32_t* z = row + size_t(a) * kPols * kPols * 2;
                // divided by the valid samples of the product (the weights'
                // correlation plus the complete gulps before the first lossy one)
                float f = inv;
                if (wtd) {
                  const double nw = double(wbase) + wrow[size_t(a) * kPols * kPols * 2];
                  f = nw > 0 ? float(1.0 / nw) : 0.f;
                }
                std::complex<float> r{float(z[0]) * f, float(z[1]) * f};
                out[size_t(a) * kStands + b] = r;
                out[size_t(b) * kStands + a] = std::conj(r);
              }
            }
          }
        }
      });
      VLOG(1) << "TccTap: published version " << v << " (" << h.ngulps << " gulps, " << h.nsamp
              << " samples, chan0 " << h.chan0 << ")";
      if (!opts.record_dir.empty() && v > 0) {
        // copy of the published table (header + R), read back from the shared-memory file
        const size_t bytes = sizeof(VisTableHeader) + size_t(nchan) * kPols * kStands * kStands * sizeof(std::complex<float>);
        char fn[1024];
        std::snprintf(fn, sizeof(fn), "%s/vis_%06llu.bin", opts.record_dir.c_str(), (unsigned long long)v);
        FILE* in = std::fopen(opts.path.c_str(), "rb");
        FILE* out = std::fopen(fn, "wb");
        if (in && out) {
          std::vector<char> buf(1 << 24);
          size_t left = bytes;
          while (left > 0) {
            const size_t k = std::fread(buf.data(), 1, std::min(left, buf.size()), in);
            if (k == 0) break;
            std::fwrite(buf.data(), 1, k, out);
            left -= k;
          }
        } else {
          LOG(ERROR) << "TccTap: cannot record table to " << fn;
        }
        if (in) std::fclose(in);
        if (out) std::fclose(out);
        LOG(INFO) << "TccTap: table " << v << " (" << h.ngulps << " gulps, nskipped " << h.nskipped << ") -> " << fn;
      }
      std::lock_guard<std::mutex> lk(mtx);
      busy = false;
    }
  }
};

TccTap::TccTap(TccTapOptions p_opts, int p_device, int p_nchan, int p_nseq, int p_nstreams)
    : m(std::make_unique<Impl>(std::move(p_opts), p_device, p_nchan, p_nseq, p_nstreams)) {}
TccTap::~TccTap() = default;
bool TccTap::Arm(uint64_t p_seq_start, int64_t p_chan0, int p_nchan, int p_nseq, float p_quality) {
  return m->Arm(p_seq_start, p_chan0, p_nchan, p_nseq, p_quality);
}
void TccTap::BeforeCopy(cudaStream_t p_stream) { Check(cudaStreamWaitEvent(p_stream, m->ev_read, 0), "wait read"); }
void TccTap::AfterCopy(int p_i, cudaStream_t p_stream) {
  if (m->armed) Check(cudaEventRecord(m->ev_copied[p_i], p_stream), "record copy");
}
void TccTap::Launch(const uint8_t* p_gulp) { m->Launch(p_gulp); }
void TccTap::SetGulpValidity(const uint64_t* p_mask, size_t p_words, const uint8_t* p_block) {
  m->SetGulpValidity(p_mask, p_words, p_block);
}
uint64_t TccTap::Version() const { return m->writer.Version(); }

#else  // !EPIC_WITH_TCC

struct TccTap::Impl {};
TccTap::TccTap(TccTapOptions, int, int, int, int) {
  LOG(FATAL) << "--caltap_gpu needs epic++ built with the Tensor-Core Correlator (cmake -DEPIC_TCC_ROOT=...)";
}
TccTap::~TccTap() = default;
bool TccTap::Arm(uint64_t, int64_t, int, int, float) { return false; }
void TccTap::BeforeCopy(cudaStream_t) {}
void TccTap::AfterCopy(int, cudaStream_t) {}
void TccTap::Launch(const uint8_t*) {}
void TccTap::SetGulpValidity(const uint64_t*, size_t, const uint8_t*) {}
uint64_t TccTap::Version() const { return 0; }

#endif  // EPIC_WITH_TCC
