// 256 x 256 EPIC images on an RTX 4090: two exact routes, checked against each
// other and against the production 128 x 128 kernel, and timed.
//
// The 4090 cannot hold a 256 x 256 dual-pol grid in one thread block (512 KB
// of half-precision grid against 256 KB of registers and 100 KB of shared
// memory per SM), so BlockEpicImager cannot simply be built for 256.
//
// 1. Polyphase EPIC. The gridded aperture occupies grid cells 0..127, so its
//    256-point transform splits into four 128-point ones:
//      E256(2k + r) = sum_n G(n) exp(i pi r n / 128) exp(2 pi i k n / 128),
//    r = (rx, ry) in {0, 1}^2. Each part is the production kernel with the
//    kernel weights of every cell multiplied by exp(i pi (rx x + ry y) / 128),
//    i.e. the image shifted by half a pixel. The auto-correlation kernels of
//    the ramped weights carry the same ramp, so --remove_autocorr still works.
//    Cost: four production launches per gulp.
//
// 2. FX. Summed over a gulp, an EPIC image is the transform of the gridded
//    aperture's auto-correlation, which is the visibilities gridded with each
//    pair's kernel cross-correlation:
//      I(k) = sum_D L(D) exp(2 pi i k D / N),
//      L(D) = sum_ab g_a g_b* R_ab sum_{c_a,i - c_b,j = D} K_a(i) K_b*(j),
//    R_ab = sum_t V_a V_b*. Leaving out a = b is --remove_autocorr. The
//    visibilities come from the Tensor-Core Correlator (libtcc); the lag grid
//    is filled once per image and transformed with cuFFT. N = 128 with lags
//    wrapped modulo 128 reproduces the production kernel exactly, which checks
//    the conventions; N = 256 gives the unaliased image.
//
// usage: fx_imager [--check | --time] [--nseq N] [--nchan C] [--chan0 c0]
//                  [--wproj 0|1] [--remove-autocorr 0|1] [--accum16 0|1]
//                  [--gulps G] [--gulps-per-image 2]
#include <cuda.h>
#include <cufft.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "epic_harness.cuh"
#include "libtcc/Correlator.h"

using namespace epic_test;

namespace {

constexpr int kStands = LWA_SV_NSTANDS;
constexpr int kInputs = kStands * LWA_SV_NPOLS;
constexpr int kT = 16;  // TCC i8 samples per block
constexpr int kNbl = kStands * (kStands + 1) / 2;

#define CUFFT_CHECK(x)                                                          \
  do {                                                                          \
    cufftResult r__ = (x);                                                      \
    if (r__ != CUFFT_SUCCESS)                                                   \
      throw std::runtime_error(std::string(#x) + ": cufft error " + std::to_string(int(r__))); \
  } while (0)

// [chan][seq][input] CNib bytes (re high nibble, im low) -> TCC i8
// [chan][seq / 16][input][16][re, im] (from src/ex/tcc_tap.cu)
__global__ void ReorderToTccI8(const uint8_t* __restrict__ in, int8_t* __restrict__ out, int nseq, int ntb) {
  const int i = threadIdx.x, tb = blockIdx.x, c = blockIdx.y;
  const uint8_t* src = in + size_t(c) * nseq * kInputs + i;
  alignas(16) int8_t v[2 * kT];
#pragma unroll
  for (int tt = 0; tt < kT; ++tt) {
    const int t = tb * kT + tt;
    const uint8_t b = t < nseq ? src[size_t(t) * kInputs] : 0;
    v[2 * tt] = int8_t(b) >> 4;
    v[2 * tt + 1] = int8_t(uint8_t(b << 4)) >> 4;
  }
  uint4* dst = reinterpret_cast<uint4*>(out + ((size_t(c) * ntb + tb) * kInputs + i) * 2 * kT);
  dst[0] = reinterpret_cast<const uint4*>(v)[0];
  dst[1] = reinterpret_cast<const uint4*>(v)[1];
}

__device__ inline float2 cmul(float2 a, float2 b) { return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }
__device__ inline float2 cmulc(float2 a, float2 b) { return {a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y}; }  // a conj(b)

// Lag grids L+[chan][XX, YY, XY, YX][N][N] from the a <= b visibilities.
// Block: (2S - 1)^2 threads (one lag offset each) by kPairs pairs; grid
// (pairs / kPairs, chan). base: cell of each stand's (0, 0) kernel element
// (int(pos)), K: complex weights [chan][stand][S^2], g: gains.
// TCC visibilities [chan][bl][polY][polX], bl = x (x + 1) / 2 + y, y <= x;
// tcc_conj: TCC holds sum s_x conj(s_y) and must be conjugated to give
// R_yx = sum V_y conj(V_x).
constexpr int kPairs = 4;
template <int S>
__global__ void GridLags(const int2* __restrict__ vis, const int2* __restrict__ base,
                         const float2* __restrict__ K, const float4* __restrict__ g,
                         float2* __restrict__ lag, int N, bool keep_auto, bool tcc_conj) {
  constexpr int h = S / 2, s2 = 2 * S - 1, nel = S * S;
  const int c = blockIdx.y;
  const int bl = blockIdx.x * kPairs + threadIdx.y;
  if (bl >= kNbl) return;
  const int x = int((sqrtf(8.f * bl + 1.f) - 1.f) * 0.5f);
  int bx = x;
  if ((bx + 1) * (bx + 2) / 2 <= bl) ++bx;
  if (bx * (bx + 1) / 2 > bl) --bx;
  const int b = bx, a = bl - bx * (bx + 1) / 2;  // a <= b
  if (a == b && !keep_auto) return;
  const float4 ga = g[c * kStands + a], gb = g[c * kStands + b];
  if ((ga.x == 0.f && ga.y == 0.f && ga.z == 0.f && ga.w == 0.f) ||
      (gb.x == 0.f && gb.y == 0.f && gb.z == 0.f && gb.w == 0.f))
    return;
  const int d = threadIdx.x;
  if (d >= s2 * s2) return;
  const int ddy = d / s2 - (S - 1), ddx = d % s2 - (S - 1);
  // C_ab(d) = sum K_a(i) conj(K_b(j)) over dx_i - dx_j = ddx, dy_i - dy_j = ddy
  const float2* Ka = K + (size_t(c) * kStands + a) * nel;
  const float2* Kb = K + (size_t(c) * kStands + b) * nel;
  float2 C{0.f, 0.f};
  for (int dyi = max(-h, ddy - h); dyi <= min(h, ddy + h); ++dyi) {
    const int dyj = dyi - ddy;
    for (int dxi = max(-h, ddx - h); dxi <= min(h, ddx + h); ++dxi) {
      const int dxj = dxi - ddx;
      const float2 ka = Ka[(h - dyi) * S + dxi + h], kb = Kb[(h - dyj) * S + dxj + h];
      C.x += ka.x * kb.x + ka.y * kb.y;
      C.y += ka.y * kb.x - ka.x * kb.y;
    }
  }
  if (C.x == 0.f && C.y == 0.f) return;
  if (a == b) C = {0.5f * C.x, 0.5f * C.y};
  const int2 pa = base[c * kStands + a], pb = base[c * kStands + b];
  int lx = (pa.x - pb.x + ddx) % N, ly = (pa.y - pb.y + ddy) % N;
  lx += lx < 0 ? N : 0;
  ly += ly < 0 ? N : 0;
  const int2* v = vis + (size_t(c) * kNbl + bl) * 4;  // [polY][polX]
  const float2 gax{ga.x, ga.y}, gay{ga.z, ga.w}, gbx{gb.x, gb.y}, gby{gb.z, gb.w};
  // products p, q (stand a pol p, stand b pol q): XX, YY, XY, YX
  const int pp[4] = {0, 1, 0, 1}, qq[4] = {0, 1, 1, 0};
  float2* out = lag + (size_t(c) * 4 * N + ly) * N + lx;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int2 r = v[pp[k] * 2 + qq[k]];
    const float2 R{float(r.x), tcc_conj ? -float(r.y) : float(r.y)};
    const float2 gp = pp[k] ? gay : gax, gq = qq[k] ? gby : gbx;
    const float2 t = cmul(cmulc(gp, gq), cmul(R, C));
    atomicAdd(&out[size_t(k) * N * N].x, t.x);
    atomicAdd(&out[size_t(k) * N * N].y, t.y);
  }
}

// Transformed lag grids F[chan][4][ky][kx] -> EPIC layout out[chan][kx][ky][XX, YY, Re XY*, Im XY*]
//   XX = 2 Re F(L+xx), YY = 2 Re F(L+yy), XY* = F(L+xy) + conj F(L+yx),
// scaled like BlockEpicImager (|E / 64|^2 x 128 per sequence).
__global__ void CombineImages(const float2* __restrict__ F, float4* __restrict__ out, int N, float scale) {
  const int c = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;  // ky * N + kx
  if (i >= N * N) return;
  const int ky = i / N, kx = i % N;
  const size_t nn = size_t(N) * N, o = size_t(c) * 4 * nn + i;
  const float2 xx = F[o], yy = F[o + nn], xy = F[o + 2 * nn], yx = F[o + 3 * nn];
  out[size_t(c) * nn + size_t(kx) * N + ky] =
      float4{2.f * xx.x * scale, 2.f * yy.x * scale, (xy.x + yx.x) * scale, (xy.y - yx.y) * scale};
}

struct Args {
  bool check = true;
  int nseq = 64, nchan = 16, chan0 = 3104, gulps = 50, gulps_per_image = 2;
  bool wproj = true, remove_autocorr = true, accum16 = true;
};

Args Parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&] { return std::atoi(argv[++i]); };
    if (k == "--check") a.check = true;
    else if (k == "--time") a.check = false;
    else if (k == "--nseq") a.nseq = next();
    else if (k == "--nchan") a.nchan = next();
    else if (k == "--chan0") a.chan0 = next();
    else if (k == "--gulps") a.gulps = next();
    else if (k == "--gulps-per-image") a.gulps_per_image = next();
    else if (k == "--wproj") a.wproj = next();
    else if (k == "--remove-autocorr") a.remove_autocorr = next();
    else if (k == "--accum16") a.accum16 = next();
    else throw std::invalid_argument("unknown option " + k);
  }
  return a;
}

// Kernel weights with the half-pixel ramp of polyphase part (rx, ry)
std::vector<float> Ramp(const ImagerHarness& h, const std::vector<float>& cplx, int rx, int ry) {
  const auto& cfg = h.config();
  const int S = cfg.support, half = S / 2, nel = S * S;
  const auto& pos = h.host_antpos();
  std::vector<float> out(cplx);
  for (int c = 0; c < cfg.nchan; ++c)
    for (int a = 0; a < kStands; ++a) {
      const float ax = pos[(size_t(c) * kStands + a) * 3], ay = pos[(size_t(c) * kStands + a) * 3 + 1];
      for (int e = 0; e < nel; ++e) {
        const int cx = int(ax + float(e % S - half)), cy = int(ay + float(half - e / S));
        const std::complex<double> w(cplx[((size_t(c) * kStands + a) * nel + e) * 2],
                                     cplx[((size_t(c) * kStands + a) * nel + e) * 2 + 1]);
        const auto r = w * std::polar(1.0, M_PI * (rx * cx + ry * cy) / ImagerHarness::N);
        out[((size_t(c) * kStands + a) * nel + e) * 2] = float(r.real());
        out[((size_t(c) * kStands + a) * nel + e) * 2 + 1] = float(r.imag());
      }
    }
  return out;
}

// The polyphase 256 x 256 EPIC imager: four production launches per gulp
struct Polyphase {
  ImagerHarness& h;
  float* w[4]{};
  float2* ac[4]{};
  float* out[4]{};
  cudaStream_t s[4]{};
  Polyphase(ImagerHarness& hh, const std::vector<float>& cplx, bool remove_autocorr) : h(hh) {
    const size_t nout = size_t(h.config().nchan) * ImagerHarness::N * ImagerHarness::N * 4;
    for (int r = 0; r < 4; ++r) {
      auto ramped = Ramp(h, cplx, r & 1, r >> 1);
      w[r] = h.UploadComplex(ramped);
      ac[r] = remove_autocorr ? h.AcorrFor(ramped) : nullptr;
      EPIC_CHECK(cudaMalloc(&out[r], nout * sizeof(float)));
      EPIC_CHECK(cudaStreamCreate(&s[r]));
    }
  }
  ~Polyphase() {
    for (int r = 0; r < 4; ++r) {
      cudaFree(w[r]);
      if (ac[r]) cudaFree(ac[r]);
      cudaFree(out[r]);
      cudaStreamDestroy(s[r]);
    }
  }
  void Launch(bool first) {
    for (int r = 0; r < 4; ++r) h.LaunchAllWith(s[r], w[r], ac[r], out[r], first);
  }
  void Sync() {
    for (auto st : s) EPIC_CHECK(cudaStreamSynchronize(st));
  }
  // [chan][Kx][Ky][4] with K = 2 k + r
  std::vector<float> Download() const {
    const int n = ImagerHarness::N, M = 2 * n, nchan = h.config().nchan;
    std::vector<float> part(size_t(nchan) * n * n * 4), img(size_t(nchan) * M * M * 4);
    for (int r = 0; r < 4; ++r) {
      const int rx = r & 1, ry = r >> 1;
      EPIC_CHECK(cudaMemcpy(part.data(), out[r], part.size() * sizeof(float), cudaMemcpyDeviceToHost));
      for (int c = 0; c < nchan; ++c)
        for (int kx = 0; kx < n; ++kx)
          for (int ky = 0; ky < n; ++ky)
            for (int p = 0; p < 4; ++p)
              img[((size_t(c) * M + 2 * kx + rx) * M + 2 * ky + ry) * 4 + p] =
                  part[((size_t(c) * n + kx) * n + ky) * 4 + p];
    }
    return img;
  }
};

// FX imager: TCC correlation per gulp; lag gridding, cuFFT and combination per image
struct FxImager {
  int N, nchan, nseq, nseq_pad, S;
  bool keep_auto, tcc_conj = false;
  std::unique_ptr<tcc::Correlator> corr;
  int8_t* d_tcc_in{};
  int2* d_vis{};
  int2* d_base{};
  float2* d_K{};
  float4* d_g{};
  float2* d_lag{};
  float4* d_out{};
  cufftHandle plan{};

  FxImager(const ImagerHarness& h, const std::vector<float>& cplx, int n, bool remove_autocorr)
      : N(n), nchan(h.config().nchan), nseq(h.config().nseq), S(h.config().support), keep_auto(!remove_autocorr) {
    nseq_pad = (nseq + kT - 1) / kT * kT;
    EPIC_CHECK(cudaFree(nullptr));
    cu::init();
    cu::Device dev(0);
    corr = std::make_unique<tcc::Correlator>(dev, tcc::i8, kStands, nchan, nseq_pad, 2);
    EPIC_CHECK(cudaMalloc(&d_tcc_in, size_t(nchan) * nseq_pad * kInputs * 2));
    EPIC_CHECK(cudaMalloc(&d_vis, size_t(nchan) * kNbl * 4 * sizeof(int2)));
    // stand cells, weights, gains
    const auto& pos = h.host_antpos();
    std::vector<int2> base(size_t(nchan) * kStands);
    for (size_t i = 0; i < base.size(); ++i) base[i] = int2{int(pos[i * 3]), int(pos[i * 3 + 1])};
    EPIC_CHECK(cudaMalloc(&d_base, base.size() * sizeof(int2)));
    EPIC_CHECK(cudaMemcpy(d_base, base.data(), base.size() * sizeof(int2), cudaMemcpyHostToDevice));
    EPIC_CHECK(cudaMalloc(&d_K, cplx.size() * sizeof(float)));
    EPIC_CHECK(cudaMemcpy(d_K, cplx.data(), cplx.size() * sizeof(float), cudaMemcpyHostToDevice));
    auto ph = h.Phases();
    EPIC_CHECK(cudaMalloc(&d_g, ph.size() * sizeof(float)));
    EPIC_CHECK(cudaMemcpy(d_g, ph.data(), ph.size() * sizeof(float), cudaMemcpyHostToDevice));
    EPIC_CHECK(cudaMalloc(&d_lag, size_t(nchan) * 4 * N * N * sizeof(float2)));
    EPIC_CHECK(cudaMalloc(&d_out, size_t(nchan) * N * N * sizeof(float4)));
    int dims[2] = {N, N};
    CUFFT_CHECK(cufftPlanMany(&plan, 2, dims, nullptr, 1, N * N, nullptr, 1, N * N, CUFFT_C2C, nchan * 4));
  }
  ~FxImager() {
    cufftDestroy(plan);
    cudaFree(d_tcc_in);
    cudaFree(d_vis);
    cudaFree(d_base);
    cudaFree(d_K);
    cudaFree(d_g);
    cudaFree(d_lag);
    cudaFree(d_out);
  }
  // Correlate one gulp (add: accumulate onto the previous gulps of the image)
  void Correlate(const uint8_t* d_feng, cudaStream_t s, bool add) {
    ReorderToTccI8<<<dim3(nseq_pad / kT, nchan), kInputs, 0, s>>>(d_feng, d_tcc_in, nseq, nseq_pad / kT);
    corr->launchAsync(reinterpret_cast<CUstream>(s), reinterpret_cast<CUdeviceptr>(d_vis),
                      reinterpret_cast<CUdeviceptr>(d_tcc_in), add);
  }
  // Lag grid, transform and combine the integrated visibilities
  void Image(cudaStream_t s) {
    EPIC_CHECK(cudaMemsetAsync(d_lag, 0, size_t(nchan) * 4 * N * N * sizeof(float2), s));
    const dim3 blk((2 * S - 1) * (2 * S - 1), kPairs), grd((kNbl + kPairs - 1) / kPairs, nchan);
    if (S != 5) throw std::invalid_argument("FX gridder is built for support 5");
    GridLags<5><<<grd, blk, 0, s>>>(d_vis, d_base, d_K, d_g, d_lag, N, keep_auto, tcc_conj);
    EPIC_CHECK(cudaGetLastError());
    CUFFT_CHECK(cufftSetStream(plan, s));
    CUFFT_CHECK(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_lag), reinterpret_cast<cufftComplex*>(d_lag),
                             CUFFT_INVERSE));
    CombineImages<<<dim3((N * N + 255) / 256, nchan), 256, 0, s>>>(d_lag, d_out, N, 1.f / 32.f);
    EPIC_CHECK(cudaGetLastError());
  }
  std::vector<float> Download() const {
    std::vector<float> o(size_t(nchan) * N * N * 4);
    EPIC_CHECK(cudaMemcpy(o.data(), d_out, o.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return o;
  }
  // Settle the TCC conjugation convention from one baseline computed on the host
  void Calibrate(const uint8_t* h_feng) {
    const int c = 0, y = 3, x = 17, py = 0, px = 1;
    auto nib = [](uint8_t b, bool re) { return re ? int(int8_t(b) >> 4) : int(int8_t(uint8_t(b << 4)) >> 4); };
    long sr = 0, si = 0;  // sum s_x conj(s_y)
    for (int t = 0; t < nseq; ++t) {
      const uint8_t* row = h_feng + (size_t(c) * nseq + t) * kInputs;
      const uint8_t bx = row[x * 2 + px], by = row[y * 2 + py];
      const int xr = nib(bx, true), xi = nib(bx, false), yr = nib(by, true), yi = nib(by, false);
      sr += xr * yr + xi * yi;
      si += xi * yr - xr * yi;
    }
    int2 v;
    EPIC_CHECK(cudaMemcpy(&v, d_vis + (size_t(c) * kNbl + x * (x + 1) / 2 + y) * 4 + py * 2 + px, sizeof(int2),
                          cudaMemcpyDeviceToHost));
    if (v.x == sr && v.y == si) tcc_conj = true;
    else if (v.x == sr && v.y == -si) tcc_conj = false;
    else throw std::runtime_error("TCC visibility matches neither convention");
    std::printf("TCC convention: visibility = %s (checked on one baseline)\n",
                tcc_conj ? "s_recvX conj(s_recvY)" : "s_recvY conj(s_recvX)");
  }
};

struct Diff {
  double max_rel, rms_rel;
};
// |a - b| relative to max |b|, per product p
Diff Compare(const std::vector<float>& a, const std::vector<float>& b, int p) {
  double mx = 0, md = 0, sd = 0, sb = 0;
  size_t n = a.size() / 4;
  for (size_t i = 0; i < n; ++i) {
    const double va = a[i * 4 + p], vb = b[i * 4 + p];
    mx = std::max(mx, std::fabs(vb));
    md = std::max(md, std::fabs(va - vb));
    sd += (va - vb) * (va - vb);
    sb += vb * vb;
  }
  return {md / mx, std::sqrt(sd / sb)};
}

void Report(const char* what, const std::vector<float>& a, const std::vector<float>& b) {
  static const char* names[4] = {"XX", "YY", "Re XY*", "Im XY*"};
  std::printf("%-40s", what);
  for (int p = 0; p < 4; ++p) {
    auto d = Compare(a, b, p);
    std::printf("  %s max %.1e rms %.1e", names[p], d.max_rel, d.rms_rel);
  }
  std::printf("\n");
}

// Brightest pixel of channel c, XX, in EPIC layout [kx][ky]
void Peak(const std::vector<float>& img, int N, int c, int* kx, int* ky) {
  float best = -1e30f;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      float v = img[((size_t(c) * N + i) * N + j) * 4];
      if (v > best) { best = v; *kx = i; *ky = j; }
    }
}

int Check(const Args& a) {
  ImagerConfig cfg;
  cfg.nseq = a.nseq;
  cfg.nchan = a.nchan;
  cfg.nstreams = 1;
  cfg.chan0 = a.chan0;
  cfg.wproj = a.wproj;
  cfg.remove_autocorr = a.remove_autocorr;
  cfg.accum16 = a.accum16;
  ImagerHarness h(cfg);
  const auto& pos = h.host_antpos();
  float lo = 1e9f, hi = -1e9f;
  for (int c = 0; c < cfg.nchan; ++c)
    for (int s = 0; s < kStands; ++s)
      for (int k = 0; k < 2; ++k) {
        lo = std::min(lo, pos[(size_t(c) * kStands + s) * 3 + k]);
        hi = std::max(hi, pos[(size_t(c) * kStands + s) * 3 + k]);
      }
  std::printf("chan0 %d (%.2f-%.2f MHz), %d channels, %d sequences; stand cells %.1f..%.1f of 128 (support %d)\n",
              cfg.chan0, cfg.chan0 * kChanBw / 1e6, (cfg.chan0 + cfg.nchan - 1) * kChanBw / 1e6, cfg.nchan,
              cfg.nseq, lo, hi, cfg.support);
  // Two sources (one at 30 deg elevation) and noise
  const double dl = pixel_dl();
  h.SimulateSources({{20.3 * dl, -11.6 * dl, 1.0, true}, {-0.62, 0.55, 0.7, true}}, 0.6);
  h.UploadFEng();
  auto real = h.GcfElements();
  auto cplx = a.wproj ? WprojElements(h, real) : RealAsComplex(real);
  h.SetComplexElements(cplx);

  // production 128 x 128
  h.LaunchAll(h.stream(0), true);
  h.Sync();
  auto epic128 = h.DownloadOut();

  Polyphase poly(h, cplx, a.remove_autocorr);
  poly.Launch(true);
  poly.Sync();
  auto epic256 = poly.Download();

  FxImager fx128(h, cplx, 128, a.remove_autocorr), fx256(h, cplx, 256, a.remove_autocorr);
  fx128.Correlate(h.device_f_eng(), 0, false);
  EPIC_CHECK(cudaDeviceSynchronize());
  fx128.Calibrate(h.host_f_eng());
  fx256.tcc_conj = fx128.tcc_conj;
  fx128.Image(0);
  fx256.Correlate(h.device_f_eng(), 0, false);
  fx256.Image(0);
  EPIC_CHECK(cudaDeviceSynchronize());
  auto f128 = fx128.Download(), f256 = fx256.Download();

  std::printf("relative differences (max over pixels and channels, and rms; relative to the reference's max):\n");
  Report("FX 128 (wrapped lags) vs production 128", f128, epic128);
  Report("polyphase EPIC 256 vs FX 256", epic256, f256);
  // The even pixels of the 256 image are the 128 image only without aliasing
  std::vector<float> even(epic128.size());
  for (int c = 0; c < cfg.nchan; ++c)
    for (int i = 0; i < 128; ++i)
      for (int j = 0; j < 128; ++j)
        for (int p = 0; p < 4; ++p)
          even[((size_t(c) * 128 + i) * 128 + j) * 4 + p] = f256[((size_t(c) * 256 + 2 * i) * 256 + 2 * j) * 4 + p];
  Report("FX 256 even pixels vs production 128", even, epic128);
  int px, py, qx, qy;
  Peak(epic128, 128, cfg.nchan - 1, &px, &py);
  Peak(f256, 256, cfg.nchan - 1, &qx, &qy);
  std::printf("brightest pixel, last channel: 128 image (%d, %d), 256 image (%d, %d)\n", px, py, qx, qy);
  return 0;
}

double Ms(cudaEvent_t a, cudaEvent_t b) {
  float ms;
  EPIC_CHECK(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

struct Stat {
  double mean, p99, max;
};
Stat Summ(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  double s = 0;
  for (double x : v) s += x;
  return {s / v.size(), v[std::min(v.size() - 1, size_t(0.99 * (v.size() - 1) + 0.5))], v.back()};
}

int Time(const Args& a) {
  ImagerConfig cfg;
  cfg.nseq = a.nseq;
  cfg.nchan = a.nchan;
  cfg.nstreams = 1;
  cfg.chan0 = a.chan0;
  cfg.wproj = a.wproj;
  cfg.remove_autocorr = a.remove_autocorr;
  cfg.accum16 = a.accum16;
  ImagerHarness h(cfg);
  h.SimulateNoise();
  h.UploadFEng();
  auto real = h.GcfElements();
  auto cplx = a.wproj ? WprojElements(h, real) : RealAsComplex(real);
  h.SetComplexElements(cplx);
  const double data_ms = a.nseq * SAMPLING_LEN_uS * 1e-3;
  const int gpi = a.gulps_per_image;
  std::printf("GPU timing, %d channels x %d sequences per gulp (%.2f ms of data), %d gulps per image, wproj %d, "
              "autocorr %s\n",
              cfg.nchan, cfg.nseq, data_ms, gpi, int(a.wproj), a.remove_autocorr ? "removed" : "kept");
  cudaStream_t s = h.stream(0);
  uint8_t* d_feng = const_cast<uint8_t*>(h.device_f_eng());
  cudaEvent_t e0, e1;
  EPIC_CHECK(cudaEventCreate(&e0));
  EPIC_CHECK(cudaEventCreate(&e1));
  const int warm = 4;
  auto h2d = [&](cudaStream_t st) {
    EPIC_CHECK(cudaMemcpyAsync(d_feng, h.host_f_eng(), h.f_eng_bytes(), cudaMemcpyHostToDevice, st));
  };
  // per-image times: the image covers gpi gulps of data
  auto run = [&](const char* name, auto&& gulp, auto&& image) {
    std::vector<double> img;
    for (int i = -warm; i < a.gulps / gpi; ++i) {
      EPIC_CHECK(cudaEventRecord(e0, s));
      for (int g = 0; g < gpi; ++g) gulp(g == 0);
      image();
      EPIC_CHECK(cudaEventRecord(e1, s));
      EPIC_CHECK(cudaEventSynchronize(e1));
      if (i >= 0) img.push_back(Ms(e0, e1));
    }
    auto st = Summ(img);
    const double budget = gpi * data_ms;
    std::printf("%-34s per image of %.1f ms: mean %7.2f  p99 %7.2f  max %7.2f ms  load %5.1f%%  %s\n", name, budget,
                st.mean, st.p99, st.max, 100 * st.p99 / budget, st.p99 < budget ? "REAL TIME" : "too slow");
  };

  run("production EPIC 128", [&](bool first) { h2d(s); h.LaunchAll(s, first); }, [] {});
  {
    Polyphase poly(h, cplx, a.remove_autocorr);
    cudaEvent_t ev;
    EPIC_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    run("polyphase EPIC 256 (4 launches)",
        [&](bool first) {
          h2d(s);
          EPIC_CHECK(cudaEventRecord(ev, s));
          for (int r = 0; r < 4; ++r) EPIC_CHECK(cudaStreamWaitEvent(poly.s[r], ev, 0));
          poly.Launch(first);
          for (int r = 0; r < 4; ++r) {
            EPIC_CHECK(cudaEventRecord(ev, poly.s[r]));
            EPIC_CHECK(cudaStreamWaitEvent(s, ev, 0));
          }
        },
        [] {});
    cudaEventDestroy(ev);
  }
  for (int n : {128, 256}) {
    FxImager fx(h, cplx, n, a.remove_autocorr);
    std::string name = "FX " + std::to_string(n) + " (TCC + lag grid + cuFFT)";
    run(name.c_str(), [&](bool first) { h2d(s); fx.Correlate(d_feng, s, !first); }, [&] { fx.Image(s); });
    // split: correlation per gulp, imaging per image
    std::vector<double> tc, ti;
    for (int i = 0; i < 20; ++i) {
      EPIC_CHECK(cudaEventRecord(e0, s));
      fx.Correlate(d_feng, s, false);
      EPIC_CHECK(cudaEventRecord(e1, s));
      EPIC_CHECK(cudaEventSynchronize(e1));
      tc.push_back(Ms(e0, e1));
      EPIC_CHECK(cudaEventRecord(e0, s));
      fx.Image(s);
      EPIC_CHECK(cudaEventRecord(e1, s));
      EPIC_CHECK(cudaEventSynchronize(e1));
      ti.push_back(Ms(e0, e1));
    }
    std::printf("    FX %d split: correlate (reorder + TCC) %.2f ms per gulp, lag grid + FFT + combine %.2f ms per "
                "image\n",
                n, Summ(tc).mean, Summ(ti).mean);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args a = Parse(argc, argv);
    return a.check ? Check(a) : Time(a);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
