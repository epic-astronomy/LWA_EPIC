// FX imaging on the GPU; see fx_imager.h. The kernels are those of
// tests/gpu/fx_imager.cu, which checks them against the E-field imager.
#include "./fx_imager.h"

#include <cufft.h>
#include <glog/logging.h>

#include <stdexcept>
#include <string>

#include "./constants.h"

#ifdef EPIC_WITH_TCC
#include "libtcc/Correlator.h"

namespace {

constexpr int kStands = LWA_SV_NSTANDS;
constexpr int kInputs = kStands * LWA_SV_NPOLS;
constexpr int kT = 16;  // TCC i8 samples per block
constexpr int kNbl = kStands * (kStands + 1) / 2;
constexpr int kPairs = 4;
constexpr int kSupport = 5;

void Check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("FxImager: ") + what + ": " + cudaGetErrorString(e));
}
void CheckFft(cufftResult r, const char* what) {
  if (r != CUFFT_SUCCESS) throw std::runtime_error(std::string("FxImager: ") + what + ": cufft error " + std::to_string(int(r)));
}

// [chan][seq][input] CNib bytes (re high nibble, im low) -> TCC i8
// [chan][seq / 16][input][16][re, im]
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

// Stand cells (int of the kernel centre position) from [chan][stand][3] positions
__global__ void BaseCells(const float* __restrict__ antpos, int2* __restrict__ base, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) base[i] = int2{int(antpos[size_t(i) * 3]), int(antpos[size_t(i) * 3 + 1])};
}

// Real kernel weights as complex
__global__ void RealToComplex(const float* __restrict__ re, float2* __restrict__ out, size_t n) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = float2{re[i], 0.f};
}

__device__ inline float2 cmul(float2 a, float2 b) { return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }
__device__ inline float2 cmulc(float2 a, float2 b) { return {a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y}; }  // a conj(b)

// Lag grids L+[chan][XX, YY, XY, YX][N][N] from the a <= b visibilities.
// Block: (2S - 1)^2 threads (one lag offset each) by kPairs pairs; grid
// (pairs / kPairs, chan). TCC visibilities [chan][bl][polY][polX],
// bl = x (x + 1) / 2 + y, y <= x, holding sum s_y conj(s_x) (checked in
// tests/gpu/fx_imager.cu), i.e. R_yx = sum V_y conj(V_x) as used here.
template <int S>
__global__ void GridLags(const int2* __restrict__ vis, const int2* __restrict__ base,
                         const float2* __restrict__ K, const float4* __restrict__ g,
                         float2* __restrict__ lag, int N, bool keep_auto) {
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
    const float2 R{float(r.x), float(r.y)};
    const float2 gp = pp[k] ? gay : gax, gq = qq[k] ? gby : gbx;
    const float2 t = cmul(cmulc(gp, gq), cmul(R, C));
    atomicAdd(&out[size_t(k) * N * N].x, t.x);
    atomicAdd(&out[size_t(k) * N * N].y, t.y);
  }
}

// Transformed lag grids F[chan][4][ky][kx] -> out[chan][kx][ky][XX, YY, Re XY*, Im XY*]:
//   XX = 2 Re F(L+xx), YY = 2 Re F(L+yy), XY* = F(L+xy) + conj F(L+yx),
// scaled like the E-field imager and multiplied by the gridding correction
// corr[chan][kx][ky] (the E-field imager's index) when given.
__global__ void CombineImages(const float2* __restrict__ F, float4* __restrict__ out,
                              const float* __restrict__ corr, int N, float scale) {
  const int c = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;  // ky * N + kx
  if (i >= N * N) return;
  const int ky = i / N, kx = i % N;
  const size_t nn = size_t(N) * N, o = size_t(c) * 4 * nn + i;
  const size_t j = size_t(c) * nn + size_t(kx) * N + ky;
  const float s = corr ? scale * corr[j] : scale;
  const float2 xx = F[o], yy = F[o + nn], xy = F[o + 2 * nn], yx = F[o + 3 * nn];
  out[j] = float4{2.f * xx.x * s, 2.f * yy.x * s, (xy.x + yx.x) * s, (xy.y - yx.y) * s};
}

}  // namespace

struct FxImager::Impl {
  int device, nchan, nseq, nseq_pad, N, S;
  bool keep_auto;
  std::unique_ptr<tcc::Correlator> corr;
  int8_t* d_tcc_in{};
  int2* d_vis{};
  int2* d_base{};
  float2* d_K{};
  float* d_corr{};
  float2* d_lag{};
  cufftHandle plan{};
  cudaStream_t s{};
  cudaEvent_t ev_free{}, ev_done{};
  bool geometry{false};

  Impl(int p_device, int p_nchan, int p_nseq, int p_n, int p_support, bool p_remove_autocorr)
      : device(p_device), nchan(p_nchan), nseq(p_nseq), N(p_n), S(p_support), keep_auto(!p_remove_autocorr) {
    if (S != kSupport) throw std::invalid_argument("FxImager: built for support 5");
    if (N < 64 || (N & (N - 1))) throw std::invalid_argument("FxImager: image size must be a power of two");
    Check(cudaSetDevice(device), "device");
    nseq_pad = (nseq + kT - 1) / kT * kT;
    cu::init();
    cu::Device dev(device);
    corr = std::make_unique<tcc::Correlator>(dev, tcc::i8, kStands, nchan, nseq_pad, LWA_SV_NPOLS);
    Check(cudaMalloc(&d_tcc_in, size_t(nchan) * nseq_pad * kInputs * 2), "input");
    Check(cudaMalloc(&d_vis, size_t(nchan) * kNbl * 4 * sizeof(int2)), "visibilities");
    Check(cudaMalloc(&d_base, size_t(nchan) * kStands * sizeof(int2)), "cells");
    Check(cudaMalloc(&d_K, size_t(nchan) * kStands * S * S * sizeof(float2)), "weights");
    Check(cudaMalloc(&d_lag, size_t(nchan) * 4 * N * N * sizeof(float2)), "lag grid");
    int dims[2] = {N, N};
    CheckFft(cufftPlanMany(&plan, 2, dims, nullptr, 1, N * N, nullptr, 1, N * N, CUFFT_C2C, nchan * 4), "plan");
    Check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    CheckFft(cufftSetStream(plan, s), "stream");
    Check(cudaEventCreateWithFlags(&ev_free, cudaEventDisableTiming), "event");
    Check(cudaEventCreateWithFlags(&ev_done, cudaEventDisableTiming), "event");
    LOG(INFO) << "FxImager: " << N << " x " << N << " images of " << nchan << " channels from the Tensor-Core "
              << "Correlator (" << nseq << " sequences per gulp), auto-correlations "
              << (keep_auto ? "kept" : "removed") << "; "
              << (size_t(nchan) * (kNbl * 4 * 8 + 4 * N * N * 8 + nseq_pad * kInputs * 2) >> 20) << " MiB on GPU "
              << device;
  }
  ~Impl() {
    cudaSetDevice(device);
    cudaStreamSynchronize(s);
    cufftDestroy(plan);
    for (void* p : {static_cast<void*>(d_tcc_in), static_cast<void*>(d_vis), static_cast<void*>(d_base),
                    static_cast<void*>(d_K), static_cast<void*>(d_corr), static_cast<void*>(d_lag)})
      if (p) cudaFree(p);
    cudaEventDestroy(ev_free);
    cudaEventDestroy(ev_done);
    cudaStreamDestroy(s);
  }
};

FxImager::FxImager(int p_device, int p_nchan, int p_nseq, int p_n, int p_support, bool p_remove_autocorr)
    : m(std::make_unique<Impl>(p_device, p_nchan, p_nseq, p_n, p_support, p_remove_autocorr)) {}
FxImager::~FxImager() = default;

void FxImager::SetGeometry(const float* p_antpos, const float* p_gcf_real, const float* p_gcf_cplx,
                           const float* p_correction) {
  Check(cudaSetDevice(m->device), "device");
  const int ns = m->nchan * kStands;
  BaseCells<<<(ns + 255) / 256, 256, 0, m->s>>>(p_antpos, m->d_base, ns);
  const size_t nk = size_t(ns) * m->S * m->S;
  if (p_gcf_cplx) {
    Check(cudaMemcpyAsync(m->d_K, p_gcf_cplx, nk * sizeof(float2), cudaMemcpyDeviceToDevice, m->s), "weights");
  } else {
    RealToComplex<<<(nk + 255) / 256, 256, 0, m->s>>>(p_gcf_real, m->d_K, nk);
  }
  if (p_correction) {
    const size_t nc = size_t(m->nchan) * m->N * m->N;
    if (!m->d_corr) Check(cudaMalloc(&m->d_corr, nc * sizeof(float)), "correction");
    Check(cudaMemcpyAsync(m->d_corr, p_correction, nc * sizeof(float), cudaMemcpyDeviceToDevice, m->s),
          "correction");
  } else if (m->d_corr) {
    cudaFree(m->d_corr);
    m->d_corr = nullptr;
  }
  Check(cudaStreamSynchronize(m->s), "geometry");
  Check(cudaGetLastError(), "geometry");
  m->geometry = true;
}

void FxImager::Correlate(const uint8_t* p_gulp, cudaEvent_t const* p_ready, int p_nready, bool p_add) {
  for (int i = 0; i < p_nready; ++i) Check(cudaStreamWaitEvent(m->s, p_ready[i], 0), "wait");
  const int ntb = m->nseq_pad / kT;
  ReorderToTccI8<<<dim3(ntb, m->nchan), kInputs, 0, m->s>>>(p_gulp, m->d_tcc_in, m->nseq, ntb);
  Check(cudaEventRecord(m->ev_free, m->s), "event");
  m->corr->launchAsync(reinterpret_cast<CUstream>(m->s), reinterpret_cast<CUdeviceptr>(m->d_vis),
                       reinterpret_cast<CUdeviceptr>(m->d_tcc_in), p_add);
}

void FxImager::Image(const float* p_phases, float* p_out, float* p_host) {
  if (!m->geometry) throw std::logic_error("FxImager: Image before SetGeometry");
  const int N = m->N;
  const size_t nlag = size_t(m->nchan) * 4 * N * N;
  Check(cudaMemsetAsync(m->d_lag, 0, nlag * sizeof(float2), m->s), "clear");
  const dim3 blk((2 * kSupport - 1) * (2 * kSupport - 1), kPairs), grd((kNbl + kPairs - 1) / kPairs, m->nchan);
  GridLags<kSupport><<<grd, blk, 0, m->s>>>(m->d_vis, m->d_base, m->d_K, reinterpret_cast<const float4*>(p_phases),
                                            m->d_lag, N, m->keep_auto);
  Check(cudaGetLastError(), "grid");
  CheckFft(cufftExecC2C(m->plan, reinterpret_cast<cufftComplex*>(m->d_lag),
                        reinterpret_cast<cufftComplex*>(m->d_lag), CUFFT_INVERSE),
           "transform");
  // The gridding correction is normalised to the E-field imager's transform
  // at this grid size, (N x oversample)^4 at the phase centre; the lag-grid
  // transform does not grow with N, so (128 / N)^4 keeps the scale of the
  // 128-point image at any N (a 256-point image equals the 128-point one on
  // its even pixels)
  const float r = 128.f / float(N);
  CombineImages<<<dim3((N * N + 255) / 256, m->nchan), 256, 0, m->s>>>(m->d_lag, reinterpret_cast<float4*>(p_out),
                                                                     m->d_corr, N, r * r * r * r / 32.f);
  Check(cudaGetLastError(), "combine");
  if (p_host) {
    Check(cudaMemcpyAsync(p_host, p_out, size_t(m->nchan) * N * N * 4 * sizeof(float), cudaMemcpyDeviceToHost,
                          m->s),
          "copy");
  }
  Check(cudaEventRecord(m->ev_done, m->s), "event");
}

void FxImager::Sync() { Check(cudaEventSynchronize(m->ev_done), "image"); }
cudaEvent_t FxImager::InputFree() const { return m->ev_free; }
cudaStream_t FxImager::Stream() const { return m->s; }
int FxImager::N() const { return m->N; }

#else  // built without the Tensor-Core Correlator

struct FxImager::Impl {};
FxImager::FxImager(int, int, int, int, int, bool) {
  throw std::runtime_error("FX imaging needs epic++ built with -DEPIC_TCC_ROOT (the Tensor-Core Correlator)");
}
FxImager::~FxImager() = default;
void FxImager::SetGeometry(const float*, const float*, const float*, const float*) {}
void FxImager::Correlate(const uint8_t*, cudaEvent_t const*, int, bool) {}
void FxImager::Image(const float*, float*, float*) {}
void FxImager::Sync() {}
cudaEvent_t FxImager::InputFree() const { return nullptr; }
cudaStream_t FxImager::Stream() const { return nullptr; }
int FxImager::N() const { return 0; }

#endif
