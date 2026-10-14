// Standalone harness for the epic++ imaging kernel (BlockEpicImager).
//
// Reproduces what MOFFCuHandler/MOFFCorrelator do for one GPU, using the
// production headers directly: the GCF texture, ComputeGcfElements, the
// kernel selected by GetImagingKernel, the launch arguments and the
// per-stream split of ProcessGulp. Nothing else from epic++ (RaftLib, pybind,
// glog, networking) is needed, so it builds and runs on any worker node.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "ex/fft_dx.cuh"
#include "lwasv_fixture.h"

namespace epic_test {

// Same on-chip accumulation type as MOFF_cu_handler.cu (--accum_16bit): bfloat16
// since 758d6e3 (2024-04-08); gridding and FFTs are __half (cuFFTDx Precision<half>)
typedef __nv_bfloat162 f16_accum_t;

#define EPIC_CHECK(x)                                                        \
  do {                                                                       \
    cudaError_t err__ = (x);                                                 \
    if (err__ != cudaSuccess) {                                              \
      throw std::runtime_error(std::string(#x) + ": " +                      \
                               cudaGetErrorString(err__) + " at " __FILE__  \
                               ":" + std::to_string(__LINE__));              \
    }                                                                        \
  } while (0)

constexpr double kSol = 299792458.0;
constexpr double kChanBw = 25e3;

enum class PhaseCenter { kPlane, kZenith };

// Production settings (consul alchemist_sv/configs/epic*/imager):
// --imagesize 128 --imageres 1.056 --nts 1000 --channels 128 --support 5
// --accum_16bit --nstreams 8
struct ImagerConfig {
  int nseq = 1000;
  int nchan = 128;
  int nstreams = 8;
  int support = 5;
  int chan0 = lwasv_fixture::kChan0;
  bool accum16 = true;
  bool wproj = false;
  PhaseCenter pc = PhaseCenter::kPlane;
  // GcfWeightMode, or kProduction (-1) for the kernel GetImagingKernel picks
  int weight_mode = -1;
  // Gridding correction multiplied into the image (gcf_correction_grid)
  int correction = 0;  // 0 unit, 1 legacy (oversampled average kernel), 2 exact
  int kernel_oversample = 4;  // production --kernel_oversample
  // --remove_autocorr: per-stand auto-correlation removal (acorr_kernels)
  bool remove_autocorr = false;
  // Sequences per kernel launch: each stream images its gulp in launches of
  // this many sequences, back to back (MOFFCuHandler m_seq_chunk). 0 = one
  // launch per gulp.
  int seq_chunk = 0;
};

enum CorrectionKind { kCorrUnit = 0, kCorrLegacy = 1, kCorrExact = 2 };

inline const char* WeightModeName(int m) {
  switch (m) {
    case -1: return "production";
    case kGcfGlobal: return "global";
    case kGcfGlobalHalf: return "global-half2";
    case kGcfShared: return "shared";
    case kGcfReg: return "registers";
    case kGcfRegHalf: return "registers-half2";
  }
  return "?";
}

// Imaging kernel for a weight mode. kProduction (-1) goes through
// GetImagingKernel exactly like MOFFCuHandler; the explicit modes are
// instantiated for the production support (5).
constexpr int kProduction = -1;
template <class FFT, typename Acc>
void* ImagingKernelFor(int support, int weight_mode) {
  if (weight_mode == kProduction) return GetImagingKernel<FFT, Acc>(support);
  if (support != 5) throw std::invalid_argument("explicit weight modes are built for support 5 only");
  switch (weight_mode) {
    case kGcfGlobal: return (void*)BlockEpicImager<FFT, 5, Acc, CHAN_MAJOR, kGcfGlobal>;
    case kGcfGlobalHalf: return (void*)BlockEpicImager<FFT, 5, Acc, CHAN_MAJOR, kGcfGlobalHalf>;
    case kGcfShared: return (void*)BlockEpicImager<FFT, 5, Acc, CHAN_MAJOR, kGcfShared>;
    case kGcfReg: return (void*)BlockEpicImager<FFT, 5, Acc, CHAN_MAJOR, kGcfReg>;
    case kGcfRegHalf: return (void*)BlockEpicImager<FFT, 5, Acc, CHAN_MAJOR, kGcfRegHalf>;
  }
  throw std::invalid_argument("unknown weight mode");
}

struct Source {
  // Direction cosines in the grid frame (x: grid east, y: grid north)
  double l, m;
  double amplitude;
  // Include the stand's residual w in the simulated voltages
  bool with_w;
};

inline const float* fixture_loc(PhaseCenter pc) {
  return pc == PhaseCenter::kPlane ? lwasv_fixture::kLocPlane
                                   : lwasv_fixture::kLocZenith;
}
inline const float* fixture_w(PhaseCenter pc) {
  return pc == PhaseCenter::kPlane ? lwasv_fixture::kWPlane
                                   : lwasv_fixture::kWZenith;
}

// Pixel spacing in direction cosine for the configured grid
inline double pixel_dl() {
  return 2.0 * std::sin(lwasv_fixture::kGridRes * M_PI / 180.0 / 2.0);
}

namespace detail {
inline void fft1d(std::complex<double>* a, int n, int stride, bool inverse) {
  std::vector<std::complex<double>> buf(n);
  for (int i = 0; i < n; ++i) buf[i] = a[i * stride];
  for (int i = 1, j = 0; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(buf[i], buf[j]);
  }
  for (int len = 2; len <= n; len <<= 1) {
    double ang = 2 * M_PI / len * (inverse ? 1 : -1);
    std::complex<double> wl(std::cos(ang), std::sin(ang));
    for (int i = 0; i < n; i += len) {
      std::complex<double> w(1);
      for (int j = 0; j < len / 2; ++j) {
        auto u = buf[i + j], v = buf[i + j + len / 2] * w;
        buf[i + j] = u + v;
        buf[i + j + len / 2] = u - v;
        w *= wl;
      }
    }
  }
  for (int i = 0; i < n; ++i) a[i * stride] = inverse ? buf[i] / double(n) : buf[i];
}
inline void fft2d(std::vector<std::complex<double>>& g, int n, bool inverse) {
  for (int r = 0; r < n; ++r) fft1d(&g[size_t(r) * n], n, 1, inverse);
  for (int c = 0; c < n; ++c) fft1d(&g[c], n, n, inverse);
}
}  // namespace detail

class ImagerHarness {
 public:
  static constexpr int N = 128;
  using FFT = FFT128x128;

  explicit ImagerHarness(const ImagerConfig& cfg) : cfg_(cfg) {
    if (cfg.nchan % cfg.nstreams != 0)
      throw std::invalid_argument("nchan must be a multiple of nstreams");
    nchan_per_stream_ = cfg.nchan / cfg.nstreams;
    f_eng_bytes_ = size_t(cfg.nchan) * cfg.nseq * LWA_SV_INP_PER_CHAN;
    EPIC_CHECK(cudaMalloc(&d_f_eng_, f_eng_bytes_));
    EPIC_CHECK(cudaMallocHost(&h_f_eng_, f_eng_bytes_));
    std::fill(h_f_eng_, h_f_eng_ + f_eng_bytes_, uint8_t(0));
    out_floats_ = size_t(cfg.nchan) * N * N * 4;
    EPIC_CHECK(cudaMalloc(&d_out_, out_floats_ * sizeof(float)));
    streams_.resize(cfg.nstreams);
    for (auto& s : streams_) EPIC_CHECK(cudaStreamCreate(&s));
    SetupGeometry();
    SetupGcf();
    SetupKernel();
  }

  ~ImagerHarness() {
    for (auto& s : streams_) cudaStreamDestroy(s);
    cudaFree(d_f_eng_);
    cudaFreeHost(h_f_eng_);
    cudaFree(d_out_);
    cudaFree(d_antpos_);
    cudaFree(d_phases_);
    cudaFree(d_gcf_elem_);
    cudaFree(d_corr_grid_);
    if (d_gcf_cplx_) cudaFree(d_gcf_cplx_);
    if (d_acorr_) cudaFree(d_acorr_);
    if (d_acorr_power_) cudaFree(d_acorr_power_);
    for (auto& e : graph_exec_) if (e) cudaGraphExecDestroy(e);
    if (fork_ev_) cudaEventDestroy(fork_ev_);
    for (auto& e : join_ev_) cudaEventDestroy(e);
    cudaDestroyTextureObject(gcf_tex_);
    cudaFreeArray(gcf_arr_);
  }

  const ImagerConfig& config() const { return cfg_; }
  int nchan_per_stream() const { return nchan_per_stream_; }
  size_t f_eng_bytes() const { return f_eng_bytes_; }
  uint8_t* host_f_eng() { return h_f_eng_; }
  float* device_out() { return d_out_; }
  const std::vector<float>& host_antpos() const { return h_antpos_; }

  // Real kernel elements computed by ComputeGcfElements (nchan x 256 x S^2)
  std::vector<float> GcfElements() const {
    std::vector<float> e(size_t(cfg_.nchan) * LWA_SV_NSTANDS * cfg_.support *
                         cfg_.support);
    EPIC_CHECK(cudaMemcpy(e.data(), d_gcf_elem_, e.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return e;
  }

  // Complex kernel elements (interleaved re/im); empty vector restores the real
  // path. Uploaded as half2 for kGcfGlobalHalf, float2 otherwise.
  void SetComplexElements(const std::vector<float>& cplx) {
    if (d_gcf_cplx_) {
      EPIC_CHECK(cudaFree(d_gcf_cplx_));
      d_gcf_cplx_ = nullptr;
    }
    UpdateAcorr(cplx);
    if (cplx.empty()) return;
    if (cfg_.weight_mode == kGcfGlobalHalf) {
      std::vector<__half2> h(cplx.size() / 2);
      for (size_t i = 0; i < h.size(); ++i) h[i] = __floats2half2_rn(cplx[2 * i], cplx[2 * i + 1]);
      EPIC_CHECK(cudaMalloc(&d_gcf_cplx_, h.size() * sizeof(__half2)));
      EPIC_CHECK(cudaMemcpy(d_gcf_cplx_, h.data(), h.size() * sizeof(__half2), cudaMemcpyHostToDevice));
      return;
    }
    EPIC_CHECK(cudaMalloc(&d_gcf_cplx_, cplx.size() * sizeof(float)));
    EPIC_CHECK(cudaMemcpy(d_gcf_cplx_, cplx.data(), cplx.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  // Calibration phases [chan][stand][Xre, Xim, Yre, Yim] (default: unit phases
  // for imaged stands, since simulated data carry no cable delays)
  void SetPhases(const std::vector<float>& ph) {
    if (ph.size() != size_t(cfg_.nchan) * LWA_SV_NSTANDS * 4)
      throw std::invalid_argument("phases must be nchan x 256 x 4 floats");
    EPIC_CHECK(cudaMemcpy(d_phases_, ph.data(), ph.size() * sizeof(float), cudaMemcpyHostToDevice));
  }

  // Fill host F-engine data (CHAN_MAJOR, 4-bit) with point sources and noise
  void SimulateSources(const std::vector<Source>& srcs, double noise_rms,
                       uint32_t seed = 1234) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    // Keep the coherent sum of all sources plus noise inside the 4-bit range
    // (+/-7) so the voltages are not clipped; relative fluxes are unchanged.
    double amp_sum = 0;
    for (const auto& s : srcs) amp_sum += s.amplitude;
    const double gain = std::min(1.0, 6.0 / (amp_sum + 3 * noise_rms));
    const float* loc = fixture_loc(cfg_.pc);
    const float* wres = fixture_w(cfg_.pc);
    const double delta = lwasv_fixture::kDelta;
    std::vector<std::complex<double>> geo(size_t(srcs.size()) * LWA_SV_NSTANDS);
    for (int c = 0; c < cfg_.nchan; ++c) {
      double lam = kSol / ((cfg_.chan0 + c) * kChanBw);
      for (size_t s = 0; s < srcs.size(); ++s) {
        double n = std::sqrt(std::max(0.0, 1 - srcs[s].l * srcs[s].l -
                                               srcs[s].m * srcs[s].m));
        for (int a = 0; a < LWA_SV_NSTANDS; ++a) {
          // Relative stand positions in metres in the grid frame
          // (gen_loc_lwasv positions are metres divided by delta)
          double x = loc[3 * a + 0] * delta, y = loc[3 * a + 1] * delta;
          double ph = x * srcs[s].l + y * srcs[s].m;
          if (srcs[s].with_w) ph += wres[a] * (n - 1);
          geo[s * LWA_SV_NSTANDS + a] =
              std::polar(gain * srcs[s].amplitude, 2 * M_PI * ph / lam);
        }
      }
      for (int t = 0; t < cfg_.nseq; ++t) {
        // Unit amplitude, random phase: each source's power is the same in every
        // sample, so measured fluxes do not depend on the number of samples
        std::vector<std::complex<double>> sig(srcs.size());
        std::uniform_real_distribution<double> uphase(0.0, 2 * M_PI);
        for (auto& v : sig) v = std::polar(1.0, uphase(rng));
        CNib2* row = reinterpret_cast<CNib2*>(
            h_f_eng_ + size_t(LWA_SV_INP_PER_CHAN) * (size_t(c) * cfg_.nseq + t));
        for (int a = 0; a < LWA_SV_NSTANDS; ++a) {
          std::complex<double> ex(g(rng) * gain * noise_rms, g(rng) * gain * noise_rms);
          std::complex<double> ey(g(rng) * gain * noise_rms, g(rng) * gain * noise_rms);
          for (size_t s = 0; s < srcs.size(); ++s) {
            ex += sig[s] * geo[s * LWA_SV_NSTANDS + a];
            ey += sig[s] * geo[s * LWA_SV_NSTANDS + a];
          }
          row[a].X.re = Quant4(ex.real());
          row[a].X.im = Quant4(ex.imag());
          row[a].Y.re = Quant4(ey.real());
          row[a].Y.im = Quant4(ey.imag());
        }
      }
    }
  }

  // Random 4-bit voltages (timing only)
  void SimulateNoise(uint32_t seed = 1) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> u(0, 255);
    for (size_t i = 0; i < f_eng_bytes_; ++i) h_f_eng_[i] = uint8_t(u(rng));
  }

  void UploadFEng() {
    EPIC_CHECK(cudaMemcpy(d_f_eng_, h_f_eng_, f_eng_bytes_, cudaMemcpyHostToDevice));
  }

  // One launch per stream, exactly as MOFFCuHandler::ProcessGulp. With
  // copy_h2d the stream's slice of F-engine data is copied first.
  void LaunchGulp(bool first, bool copy_h2d) {
    size_t bytes_per_stream = f_eng_bytes_ / cfg_.nstreams;
    for (int i = 0; i < cfg_.nstreams; ++i) {
      int chan_offset = i * nchan_per_stream_;
      if (copy_h2d) {
        EPIC_CHECK(cudaMemcpyAsync(d_f_eng_ + i * bytes_per_stream,
                                   h_f_eng_ + i * bytes_per_stream,
                                   bytes_per_stream, cudaMemcpyHostToDevice,
                                   streams_[i]));
      }
      LaunchChunks(streams_[i], chan_offset, nchan_per_stream_, first);
    }
  }

  // MOFFCuHandler::ProcessGulp on the last gulp of an image: each stream copies
  // its channels of the output image to (pinned) host memory
  void CopyOutAsync(float* host) {
    const size_t per_stream = size_t(nchan_per_stream_) * N * N * 4;
    for (int i = 0; i < cfg_.nstreams; ++i)
      EPIC_CHECK(cudaMemcpyAsync(host + i * per_stream, d_out_ + i * per_stream,
                                 per_stream * sizeof(float), cudaMemcpyDeviceToHost, streams_[i]));
  }
  size_t out_floats() const { return out_floats_; }

  // The whole gulp (LaunchGulp with the H2D copy, plus CopyOutAsync on the
  // last gulp of an image) captured once as a CUDA graph and replayed on s.
  // One executable graph per (first, last) variant; host_out must not change.
  void LaunchGulpGraph(cudaStream_t s, bool first, bool last, float* host_out) {
    auto& exec = graph_exec_[(first ? 2 : 0) + (last ? 1 : 0)];
    if (!exec) {
      if (!fork_ev_) {
        EPIC_CHECK(cudaEventCreateWithFlags(&fork_ev_, cudaEventDisableTiming));
        join_ev_.resize(cfg_.nstreams);
        for (auto& e : join_ev_) EPIC_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      }
      cudaGraph_t g;
      EPIC_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
      EPIC_CHECK(cudaEventRecord(fork_ev_, s));
      for (auto& st : streams_) EPIC_CHECK(cudaStreamWaitEvent(st, fork_ev_, 0));
      LaunchGulp(first, true);
      if (last) CopyOutAsync(host_out);
      for (int i = 0; i < cfg_.nstreams; ++i) {
        EPIC_CHECK(cudaEventRecord(join_ev_[i], streams_[i]));
        EPIC_CHECK(cudaStreamWaitEvent(s, join_ev_[i], 0));
      }
      EPIC_CHECK(cudaStreamEndCapture(s, &g));
      EPIC_CHECK(cudaGraphInstantiate(&exec, g, 0));
      EPIC_CHECK(cudaGraphDestroy(g));
    }
    EPIC_CHECK(cudaGraphLaunch(exec, s));
  }

  // All channels in one launch on a given stream (for kernel-only timing)
  void LaunchAll(cudaStream_t stream, bool first) {
    LaunchChunks(stream, 0, cfg_.nchan, first);
  }

  void Sync() {
    for (auto& s : streams_) EPIC_CHECK(cudaStreamSynchronize(s));
    EPIC_CHECK(cudaGetLastError());
  }

  cudaStream_t stream(int i) const { return streams_[i]; }

  // Image for one channel, pol 0 (XX), unpacked like epic_utils.save_output:
  // transpose, fftshift and flip y. Returns [y][x], east left, north up.
  std::vector<float> ImageXX(const std::vector<float>& out, int chan) const {
    std::vector<float> img(N * N);
    for (int y = 0; y < N; ++y) {
      int b = ((N - 1 - y) + N / 2) % N;
      for (int x = 0; x < N; ++x) {
        int a = (x + N / 2) % N;
        img[y * N + x] = out[(size_t(chan) * N * N + size_t(a) * N + b) * 4 + 0];
      }
    }
    return img;
  }

  std::vector<float> DownloadOut() const {
    std::vector<float> out(out_floats_);
    EPIC_CHECK(cudaMemcpy(out.data(), d_out_, out.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return out;
  }

  // Expected pixel (x, y) of a grid-frame direction, save_output convention
  static void ExpectedPixel(double l, double m, double* x, double* y) {
    *x = N / 2 - l / pixel_dl();
    *y = (N / 2 - 1) + m / pixel_dl();
  }

 private:
  static signed char Quant4(double v) {
    long r = std::lround(v);
    return static_cast<signed char>(std::max(-7L, std::min(7L, r)));
  }

  void SetupGeometry() {
    // MOFFCorrelator::ResetAntpos: half_grid + raw * (chan * BW / c)
    const float* loc = fixture_loc(cfg_.pc);
    int pitch = 3 * LWA_SV_NSTANDS;
    h_antpos_.assign(size_t(pitch) * cfg_.nchan, 0.f);
    for (int c = 0; c < cfg_.nchan; ++c) {
      double wavenumber = double((cfg_.chan0 + c) * BANDWIDTH) / double(SOL);
      for (int p = 0; p < pitch; ++p)
        h_antpos_[size_t(c) * pitch + p] =
            float(N / 2.0f + loc[p] * wavenumber);
    }
    EPIC_CHECK(cudaMalloc(&d_antpos_, h_antpos_.size() * sizeof(float)));
    EPIC_CHECK(cudaMemcpy(d_antpos_, h_antpos_.data(),
                          h_antpos_.size() * sizeof(float), cudaMemcpyHostToDevice));
    // Phases: no cable delays in simulated data, so unit phases for imaged
    // stands and zero for masked ones (gen_phases_lwasv masking)
    std::vector<float> ph(size_t(cfg_.nchan) * LWA_SV_NSTANDS * 4, 0.f);
    for (int c = 0; c < cfg_.nchan; ++c)
      for (int a = 0; a < LWA_SV_NSTANDS; ++a)
        if (lwasv_fixture::kUsed[a]) {
          ph[(size_t(c) * LWA_SV_NSTANDS + a) * 4 + 0] = 1.f;
          ph[(size_t(c) * LWA_SV_NSTANDS + a) * 4 + 2] = 1.f;
        }
    EPIC_CHECK(cudaMalloc(&d_phases_, ph.size() * sizeof(float)));
    EPIC_CHECK(cudaMemcpy(d_phases_, ph.data(), ph.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  void SetupGcf() {
    // MOFFCuHandler::ResetGcfTex
    int dim = lwasv_fixture::kGcfTexDim;
    cudaChannelFormatDesc cd = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    EPIC_CHECK(cudaMallocArray(&gcf_arr_, &cd, dim, dim));
    EPIC_CHECK(cudaMemcpy2DToArray(gcf_arr_, 0, 0, lwasv_fixture::kGcfTex,
                                   dim * sizeof(float), dim * sizeof(float), dim,
                                   cudaMemcpyHostToDevice));
    cudaResourceDesc rd{};
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = gcf_arr_;
    cudaTextureDesc td{};
    td.addressMode[0] = cudaAddressModeClamp;
    td.addressMode[1] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModeLinear;
    td.readMode = cudaReadModeElementType;
    td.normalizedCoords = 0;
    EPIC_CHECK(cudaCreateTextureObject(&gcf_tex_, &rd, &td, nullptr));

    // MOFFCuHandler::ResetGcfElem with the imaging support and delta
    int nel = cfg_.support * cfg_.support;
    EPIC_CHECK(cudaMalloc(&d_gcf_elem_, size_t(LWA_SV_NSTANDS) * cfg_.nchan * nel * sizeof(float)));
    int block = int(MAX_THREADS_PER_BLOCK / float(nel)) * nel;
    ComputeGcfElements<<<cfg_.nchan, block, LWA_SV_NSTANDS * sizeof(float)>>>(
        d_gcf_elem_, d_antpos_, cfg_.chan0, float(lwasv_fixture::kDelta), gcf_tex_,
        N, cfg_.support, LWA_SV_NSTANDS);
    EPIC_CHECK(cudaGetLastError());
    EPIC_CHECK(cudaDeviceSynchronize());

    // Gridding correction (unit by default: it does not move peaks and costs
    // the same to apply whatever its values)
    std::vector<float> corr(size_t(cfg_.nchan) * N * N, 1.f);
    if (cfg_.correction == kCorrLegacy) LegacyCorrection(corr);
    if (cfg_.correction == kCorrExact) ExactCorrection(corr);
    EPIC_CHECK(cudaMalloc(&d_corr_grid_, corr.size() * sizeof(float)));
    EPIC_CHECK(cudaMemcpy(d_corr_grid_, corr.data(), corr.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    UpdateAcorr({});
  }

  // MOFFCuHandler::UpdateAcorrKernels: per-stand weight auto-correlations from
  // the real weights, or from complex weights (host, interleaved re/im) if given
  void UpdateAcorr(const std::vector<float>& cplx) {
    if (d_acorr_) {
      EPIC_CHECK(cudaFree(d_acorr_));
      d_acorr_ = nullptr;
    }
    if (!cfg_.remove_autocorr) return;
    const int S = cfg_.support, ncell = (2 * S - 1) * (2 * S - 1);
    float2* d_c = nullptr;
    if (!cplx.empty()) {
      EPIC_CHECK(cudaMalloc(&d_c, cplx.size() * sizeof(float)));
      EPIC_CHECK(cudaMemcpy(d_c, cplx.data(), cplx.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    EPIC_CHECK(cudaMalloc(&d_acorr_, size_t(cfg_.nchan) * LWA_SV_NSTANDS * ncell * sizeof(float2)));
    if (!d_acorr_power_)
      EPIC_CHECK(cudaMalloc(&d_acorr_power_, size_t(cfg_.nchan) * LWA_SV_NSTANDS * sizeof(float4)));
    ComputeAcorrKernels<<<cfg_.nchan, 256>>>(d_gcf_elem_, d_c, S, d_acorr_, LWA_SV_NSTANDS);
    EPIC_CHECK(cudaGetLastError());
    EPIC_CHECK(cudaDeviceSynchronize());
    if (d_c) cudaFree(d_c);
  }

  // Per-stand weight auto-correlations (nchan x 256 x (2S-1)^2 complex), empty if off
 public:
  std::vector<float2> AcorrKernels() const {
    if (!d_acorr_) return {};
    const int S = cfg_.support;
    std::vector<float2> a(size_t(cfg_.nchan) * LWA_SV_NSTANDS * (2 * S - 1) * (2 * S - 1));
    EPIC_CHECK(cudaMemcpy(a.data(), d_acorr_, a.size() * sizeof(float2), cudaMemcpyDeviceToHost));
    return a;
  }

 private:

  // Port of epic_geometry.gridding_correction: 1 / |mean_a FT(K_a)|^2 relative
  // to each stand's true grid position, centre value (N * oversample)^4.
  void ExactCorrection(std::vector<float>& corr) {
    const int S = cfg_.support, nel = S * S, half = S / 2;
    auto elems = GcfElements();
    const double scale = std::pow(double(N) * cfg_.kernel_oversample, 4);
    std::vector<std::complex<double>> ex(N), ey(N);
    std::vector<std::complex<double>> k(size_t(N) * N);
    for (int c = 0; c < cfg_.nchan; ++c) {
      std::fill(k.begin(), k.end(), 0.0);
      int nlive = 0;
      for (int a = 0; a < LWA_SV_NSTANDS; ++a) {
        const float* w = &elems[(size_t(c) * LWA_SV_NSTANDS + a) * nel];
        bool live = false;
        for (int e = 0; e < nel; ++e) live |= w[e] != 0.f;
        if (!live) continue;
        ++nlive;
        float px = h_antpos_[(size_t(c) * LWA_SV_NSTANDS + a) * 3 + 0];
        float py = h_antpos_[(size_t(c) * LWA_SV_NSTANDS + a) * 3 + 1];
        for (int e = 0; e < nel; ++e) {
          if (w[e] == 0.f) continue;
          double ox = std::trunc(px + float(e % S - half)) - px;
          double oy = std::trunc(py + float(half - e / S)) - py;
          for (int i = 0; i < N; ++i) {
            int ki = i < N / 2 ? i : i - N;
            ex[i] = std::polar(1.0, 2 * M_PI * ox * ki / N);
            ey[i] = std::polar(double(w[e]), 2 * M_PI * oy * ki / N);
          }
          for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) k[size_t(i) * N + j] += ex[i] * ey[j];  // [kx][ky]
        }
      }
      double r0 = std::norm(k[0]);
      for (int i = 0; i < N * N; ++i)
        corr[size_t(c) * N * N + i] = float(scale / std::max(std::norm(k[i]) / r0, 1e-6));
    }
  }

  // Port of the previous production method: ComputeGcfElements on the
  // oversampled support with delta / oversample, ComputeAvgGriddingKernel, then
  // epic_utils.get_correction_grid.
  void LegacyCorrection(std::vector<float>& corr) {
    const int OS = cfg_.kernel_oversample, So = cfg_.support * OS, nel = So * So;
    float* d_os = nullptr;
    EPIC_CHECK(cudaMalloc(&d_os, size_t(LWA_SV_NSTANDS) * cfg_.nchan * nel * sizeof(float)));
    int block = int(MAX_THREADS_PER_BLOCK / float(nel)) * nel;
    ComputeGcfElements<<<cfg_.nchan, block, LWA_SV_NSTANDS * sizeof(float)>>>(
        d_os, d_antpos_, cfg_.chan0, float(lwasv_fixture::kDelta / OS), gcf_tex_, N, So,
        LWA_SV_NSTANDS);
    EPIC_CHECK(cudaGetLastError());
    std::vector<float> os(size_t(LWA_SV_NSTANDS) * cfg_.nchan * nel);
    EPIC_CHECK(cudaMemcpy(os.data(), d_os, os.size() * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_os);
    const int M = N * OS, off = (M - N) / 2;
    std::vector<std::complex<double>> g(size_t(M) * M);
    for (int c = 0; c < cfg_.nchan; ++c) {
      std::vector<double> ker(nel, 0.0);
      double tot = 0;
      for (int a = 0; a < LWA_SV_NSTANDS; ++a)
        for (int e = 0; e < nel; ++e) ker[e] += os[(size_t(c) * LWA_SV_NSTANDS + a) * nel + e];
      for (double v : ker) tot += v;
      std::fill(g.begin(), g.end(), 0.0);
      for (int r = 0; r < So; ++r)
        for (int q = 0; q < So; ++q) g[size_t(M / 2 + r) * M + (M / 2 + q)] = ker[r * So + q] / tot;
      detail::fft2d(g, M, true);  // numpy ifft2
      // fftshift, crop to N, transpose, fftshift, |.|^2, reciprocal
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
          int si = (i + N / 2) % N, sj = (j + N / 2) % N;  // outer fftshift
          int r = ((si + off) + M / 2) % M;                // inner fftshift + crop (row)
          int q = ((sj + off) + M / 2) % M;
          double v = std::norm(g[size_t(q) * M + r]);      // transpose: [i][j] <- [j][i]
          corr[size_t(c) * N * N + size_t(i) * N + j] = float(1.0 / v);
        }
    }
  }

  void SetupKernel() {
    // MOFFCuHandler::SetImagingKernel
    kernel_ = cfg_.accum16 ? ImagingKernelFor<FFT, f16_accum_t>(cfg_.support, cfg_.weight_mode)
                           : ImagingKernelFor<FFT, float2>(cfg_.support, cfg_.weight_mode);
    block_dim_ = FFT::block_dim;

    smem_ = int(FFT::shared_memory_size * 1.5);
    EPIC_CHECK(cudaFuncSetAttribute(kernel_, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_));
  }

  void LaunchChunks(cudaStream_t stream, int chan_offset, int nblocks, bool first) {
    const int chunk = cfg_.seq_chunk > 0 ? cfg_.seq_chunk : cfg_.nseq;
    for (int b = 0; b < cfg_.nseq; b += chunk)
      LaunchOne(stream, chan_offset, nblocks, first && b == 0, b, std::min(cfg_.nseq, b + chunk));
  }

  void LaunchOne(cudaStream_t stream, int chan_offset, int nblocks, bool first,
                 int seq_begin, int seq_end) {
    int nseq = cfg_.nseq, nchan = cfg_.nchan;
    void* args[] = {&d_f_eng_, &d_antpos_, &d_phases_, &nseq, &nchan, &gcf_tex_,
                    &d_out_, &chan_offset, &first, &d_gcf_elem_, &d_corr_grid_,
                    &d_gcf_cplx_, &d_acorr_, &seq_begin, &seq_end, &d_acorr_power_};
    EPIC_CHECK(cudaLaunchKernel(kernel_, dim3(nblocks, 1, 1), block_dim_, args, smem_, stream));
  }

  ImagerConfig cfg_;
  int nchan_per_stream_{0};
  size_t f_eng_bytes_{0};
  size_t out_floats_{0};
  uint8_t* d_f_eng_{nullptr};
  uint8_t* h_f_eng_{nullptr};
  float* d_out_{nullptr};
  float* d_antpos_{nullptr};
  float* d_phases_{nullptr};
  float* d_gcf_elem_{nullptr};
  float* d_corr_grid_{nullptr};
  float* d_gcf_cplx_{nullptr};
  float2* d_acorr_{nullptr};
  float4* d_acorr_power_{nullptr};
  cudaGraphExec_t graph_exec_[4]{};
  cudaEvent_t fork_ev_{nullptr};
  std::vector<cudaEvent_t> join_ev_;
  std::vector<float> h_antpos_;
  cudaArray_t gcf_arr_{nullptr};
  cudaTextureObject_t gcf_tex_{0};
  void* kernel_{nullptr};
  dim3 block_dim_;
  int smem_{0};
  std::vector<cudaStream_t> streams_;
};

// ---- host-side w-projection (C++ port of epic_geometry.wproj_kernel_weights)



// Complex w-projected elements from the real ones, same cells as the gridder
inline std::vector<float> WprojElements(const ImagerHarness& h,
                                        const std::vector<float>& real_elems,
                                        double w_scale = 1.0) {
  const auto& cfg = h.config();
  const int N = ImagerHarness::N, S = cfg.support, half = S / 2, nel = S * S;
  const float* wres = fixture_w(cfg.pc);
  const double dl = pixel_dl();
  std::vector<double> nm1(N * N);
  for (int r = 0; r < N; ++r)
    for (int c = 0; c < N; ++c) {
      double kr = (r < N / 2 ? r : r - N) * dl, kc = (c < N / 2 ? c : c - N) * dl;
      nm1[r * N + c] = std::sqrt(std::max(0.0, 1 - kr * kr - kc * kc)) - 1.0;
    }
  std::vector<float> out(real_elems.size() * 2, 0.f);
  const auto& antpos = h.host_antpos();
  std::vector<std::complex<double>> g(N * N);
  for (int c = 0; c < cfg.nchan; ++c) {
    double lam = kSol / ((cfg.chan0 + c) * kChanBw);
    for (int a = 0; a < LWA_SV_NSTANDS; ++a) {
      const float* k = &real_elems[(size_t(c) * LWA_SV_NSTANDS + a) * nel];
      bool any = false;
      for (int e = 0; e < nel; ++e) any |= k[e] != 0.f;
      if (!any) continue;
      float ax = antpos[(size_t(c) * LWA_SV_NSTANDS + a) * 3 + 0];
      float ay = antpos[(size_t(c) * LWA_SV_NSTANDS + a) * 3 + 1];
      std::fill(g.begin(), g.end(), 0.0);
      std::vector<int> xp(nel), yp(nel);
      for (int e = 0; e < nel; ++e) {
        xp[e] = int(ax + float(e % S - half));
        yp[e] = int(ay + float(half - e / S));
        if (xp[e] >= 0 && xp[e] < N && yp[e] >= 0 && yp[e] < N) g[yp[e] * N + xp[e]] += k[e];
      }
      detail::fft2d(g, N, true);
      double wl = w_scale * wres[a] / lam;
      for (int i = 0; i < N * N; ++i) g[i] *= std::polar(1.0, -2 * M_PI * wl * nm1[i]);
      detail::fft2d(g, N, false);
      for (int e = 0; e < nel; ++e) {
        if (xp[e] < 0 || xp[e] >= N || yp[e] < 0 || yp[e] >= N) continue;
        auto v = g[yp[e] * N + xp[e]];
        size_t o = ((size_t(c) * LWA_SV_NSTANDS + a) * nel + e) * 2;
        out[o] = float(v.real());
        out[o + 1] = float(v.imag());
      }
    }
  }
  return out;
}

// Complex elements equal to the real ones (zero imaginary part)
inline std::vector<float> RealAsComplex(const std::vector<float>& real_elems) {
  std::vector<float> out(real_elems.size() * 2, 0.f);
  for (size_t i = 0; i < real_elems.size(); ++i) out[2 * i] = real_elems[i];
  return out;
}

// Sub-pixel peak near (x0, y0) by parabolic interpolation
inline bool FindPeak(const std::vector<float>& img, double x0, double y0,
                     int radius, double* px, double* py) {
  const int N = ImagerHarness::N;
  int xi = int(std::lround(x0)), yi = int(std::lround(y0));
  float best = -1e30f;
  int bx = -1, by = -1;
  for (int y = yi - radius; y <= yi + radius; ++y)
    for (int x = xi - radius; x <= xi + radius; ++x) {
      if (x < 1 || x >= N - 1 || y < 1 || y >= N - 1) continue;
      if (img[y * N + x] > best) { best = img[y * N + x]; bx = x; by = y; }
    }
  if (bx < 0) return false;
  auto par = [](double a, double b, double c) {
    double d = a - 2 * b + c;
    return d < 0 ? std::clamp(0.5 * (a - c) / d, -0.5, 0.5) : 0.0;
  };
  *px = bx + par(img[by * N + bx - 1], img[by * N + bx], img[by * N + bx + 1]);
  *py = by + par(img[(by - 1) * N + bx], img[by * N + bx], img[(by + 1) * N + bx]);
  return true;
}

}  // namespace epic_test
