// FX imaging on the GPU; see fx_imager.h. The kernels are those of
// tests/gpu/fx_imager.cu, which checks them against the E-field imager.
#include "./fx_imager.h"

#include <cuda_fp16.h>
#include <cufft.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

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

// The pair kernel table in single or half precision
__device__ __forceinline__ void StoreC(float2* p, float2 v) { *p = v; }
__device__ __forceinline__ void StoreC(__half2* p, float2 v) { *p = __float22half2_rn(v); }
__device__ __forceinline__ float2 LoadC(const float2* p) { return __ldg(p); }
__device__ __forceinline__ float2 LoadC(const __half2* p) { return __half22float2(__ldg(p)); }

// Per channel, the largest |K_a|^2 = sum_i |K_a(i)|^2 over the stands: by
// Cauchy-Schwarz it bounds every |C_ab(dd)|, so C / scale fits half precision.
template <int S>
__global__ void KernelScale(const float2* __restrict__ K, float* __restrict__ scale) {
  constexpr int nel = S * S;
  const int c = blockIdx.x, a = threadIdx.x;  // one thread per stand
  float n2 = 0.f;
  for (int i = 0; i < nel; ++i) {
    const float2 k = K[(size_t(c) * kStands + a) * nel + i];
    n2 += k.x * k.x + k.y * k.y;
  }
  for (int o = 16; o > 0; o >>= 1) n2 = fmaxf(n2, __shfl_xor_sync(0xffffffffu, n2, o));
  __shared__ float wmax[kStands / 32];
  if ((a & 31) == 0) wmax[a >> 5] = n2;
  __syncthreads();
  if (a == 0) {
    float m = 0.f;
    for (int w = 0; w < kStands / 32; ++w) m = fmaxf(m, wmax[w]);
    scale[c] = m > 0.f ? m : 1.f;
  }
}

// Kernel cross-correlations of every pair, once per geometry:
//   ctab[chan][bl][(ddy + S - 1) (2S - 1) + ddx + S - 1] = sum K_a(i) K_b*(i - dd) / scale[chan],
// halved for a = b (bl = b (b + 1) / 2 + a, a <= b).
template <int S, typename T>
__global__ void PairKernels(const float2* __restrict__ K, const float* __restrict__ scale, T* __restrict__ ctab) {
  constexpr int h = S / 2, s2 = 2 * S - 1, nel = S * S;
  const int bl = blockIdx.x, c = blockIdx.y, d = threadIdx.x;
  int b = int((sqrtf(8.f * bl + 1.f) - 1.f) / 2.f);
  while (b * (b + 1) / 2 > bl) --b;
  while ((b + 1) * (b + 2) / 2 <= bl) ++b;
  const int a = bl - b * (b + 1) / 2;
  __shared__ float2 ka[nel], kb[nel];
  if (d < nel) {
    ka[d] = K[(size_t(c) * kStands + a) * nel + d];
    kb[d] = K[(size_t(c) * kStands + b) * nel + d];
  }
  __syncthreads();
  if (d >= s2 * s2) return;
  const int ddy = d / s2 - (S - 1), ddx = d % s2 - (S - 1);
  float2 C{0.f, 0.f};
  for (int dyi = max(-h, ddy - h); dyi <= min(h, ddy + h); ++dyi) {
    const int dyj = dyi - ddy;
    for (int dxi = max(-h, ddx - h); dxi <= min(h, ddx + h); ++dxi) {
      const int dxj = dxi - ddx;
      const float2 x = ka[(h - dyi) * S + dxi + h], y = kb[(h - dyj) * S + dxj + h];
      C.x += x.x * y.x + x.y * y.y;
      C.y += x.y * y.x - x.x * y.y;
    }
  }
  const float f = (a == b ? 0.5f : 1.f) / scale[c];
  StoreC(ctab + (size_t(c) * kNbl + bl) * (s2 * s2) + d, float2{f * C.x, f * C.y});
}

// Tiled gridding: the same sums as GridLags, accumulated per lag-grid tile
// in shared memory instead of with atomics on the global grid (4 x 10^9 of
// them per image for 192 channels at 256 x 256). Block (tile, channel), eight
// warps; list[off[c][tile] ..] holds the pairs whose footprint ((2S - 1)^2
// lag cells) touches the tile, packed bl | a << 16 | b << 24. Each warp takes
// one pair at a time, its lanes the lag offsets, with the pair's kernel
// cross-correlation from ctab (PairKernels); the tile is written whole (zeros
// too), so the grid needs no clearing. A tile row's columns are rotated by
// 9 x row in shared memory: a warp's 32 lags span three to four footprint rows,
// which would otherwise fall on the same banks.
constexpr int kTile = 32;
constexpr int kTileThreads = 256;  // eight warps
__device__ __forceinline__ int TileIndex(int ux, int uy) { return uy * kTile + ((ux + 9 * uy) & (kTile - 1)); }
template <int S, typename T>
__global__ void __launch_bounds__(kTileThreads) GridLagsTiled(const int2* __restrict__ vis, const int2* __restrict__ base,
                                                      const T* __restrict__ ctab, const float* __restrict__ cscale,
                                                      const float4* __restrict__ g,
                                                      const uint32_t* __restrict__ list, const int* __restrict__ off,
                                                      float2* __restrict__ lag, int N) {
  constexpr int s2 = 2 * S - 1;
  __shared__ float sre[4][kTile * kTile];
  __shared__ float sim[4][kTile * kTile];
  const int c = blockIdx.y, tile = blockIdx.x, ntx = N / kTile;
  const int tx0 = (tile % ntx) * kTile, ty0 = (tile / ntx) * kTile;
  for (int i = threadIdx.x; i < kTile * kTile; i += blockDim.x)
#pragma unroll
    for (int k = 0; k < 4; ++k) sre[k][i] = sim[k][i] = 0.f;
  __syncthreads();
  constexpr int nd = (s2 * s2 + 31) / 32;  // lag offsets per lane
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int beg = off[c * gridDim.x + tile], end = off[c * gridDim.x + tile + 1];
  // The lanes' lag offsets relative to the footprint corner
  int ldx[nd], ldy[nd];
#pragma unroll
  for (int q = 0; q < nd; ++q) {
    const int d = q * 32 + lane;
    ldy[q] = d / s2;
    ldx[q] = d % s2;
  }
  // Warps take 32 entries at a time: every lane looks up one pair (its
  // gains, cells and visibilities), so the dependent loads overlap; the
  // pairs are then gridded one by one, broadcast with shuffles, the next
  // pair's cross-correlations loaded while the current one's are added.
  for (int e0 = beg + warp * 32; e0 < end; e0 += kTileThreads) {
    const int e = e0 + lane;
    float2 P[4] = {};
    int bl = 0, ox = 1 << 20, oy = 0;  // corner of the footprint in the tile; far away: nothing
    if (e < end) {
      const uint32_t ent = list[e];
      bl = int(ent & 0xffff);
      const int a = int((ent >> 16) & 0xff), b = int(ent >> 24);
      const float4 ga = g[c * kStands + a], gb = g[c * kStands + b];
      const int2 pa = base[c * kStands + a], pb = base[c * kStands + b];
      const int4* v = reinterpret_cast<const int4*>(vis + (size_t(c) * kNbl + bl) * 4);
      const int4 v01 = v[0], v23 = v[1];  // [polY][polX]: xx, xy | yx, yy
      if (!((ga.x == 0.f && ga.y == 0.f && ga.z == 0.f && ga.w == 0.f) ||
            (gb.x == 0.f && gb.y == 0.f && gb.z == 0.f && gb.w == 0.f))) {
        const float sc = cscale[c];
        const float2 gax{sc * ga.x, sc * ga.y}, gay{sc * ga.z, sc * ga.w}, gbx{gb.x, gb.y}, gby{gb.z, gb.w};
        P[0] = cmul(cmulc(gax, gbx), float2{float(v01.x), float(v01.y)});
        P[1] = cmul(cmulc(gay, gby), float2{float(v23.z), float(v23.w)});
        P[2] = cmul(cmulc(gax, gby), float2{float(v01.z), float(v01.w)});
        P[3] = cmul(cmulc(gay, gbx), float2{float(v23.x), float(v23.y)});
        // the footprint's corner (lag pa - pb - (S - 1)) relative to the
        // tile, in [-(2S - 2) - tile, N - tile) so lags wrap at most once
        int lx = (pa.x - pb.x - (S - 1) - tx0) % N, ly = (pa.y - pb.y - (S - 1) - ty0) % N;
        lx += lx < -(s2 - 1) ? N : 0;
        ly += ly < -(s2 - 1) ? N : 0;
        lx -= lx >= kTile ? N : 0;
        ly -= ly >= kTile ? N : 0;
        ox = lx;
        oy = ly;
      }
    }
    const int np = min(32, end - e0);
    // the cross-correlations of pair j that fall in the tile (else zero)
    auto load = [&](int j, float2* C, int* idx) {
      const int jbl = __shfl_sync(0xffffffffu, bl, j), jx = __shfl_sync(0xffffffffu, ox, j),
                jy = __shfl_sync(0xffffffffu, oy, j);
      const T* Cp = ctab + (size_t(c) * kNbl + jbl) * (s2 * s2);
#pragma unroll
      for (int q = 0; q < nd; ++q) {
        const int ux = jx + ldx[q], uy = jy + ldy[q];
        const bool in = q * 32 + lane < s2 * s2 && ux >= 0 && ux < kTile && uy >= 0 && uy < kTile;
        idx[q] = in ? TileIndex(ux, uy) : -1;
        C[q] = in ? LoadC(Cp + q * 32 + lane) : float2{0.f, 0.f};
      }
    };
    float2 C[nd], Cn[nd];
    int idx[nd], idxn[nd];
    load(0, C, idx);
    for (int j = 0; j < np; ++j) {
      if (j + 1 < np) load(j + 1, Cn, idxn);
      float2 Pj[4];
#pragma unroll
      for (int k = 0; k < 4; ++k)
        Pj[k] = float2{__shfl_sync(0xffffffffu, P[k].x, j), __shfl_sync(0xffffffffu, P[k].y, j)};
#pragma unroll
      for (int q = 0; q < nd; ++q) {
        if (idx[q] < 0 || (C[q].x == 0.f && C[q].y == 0.f)) continue;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float2 t = cmul(Pj[k], C[q]);
          atomicAdd(&sre[k][idx[q]], t.x);
          atomicAdd(&sim[k][idx[q]], t.y);
        }
      }
#pragma unroll
      for (int q = 0; q < nd; ++q) {
        C[q] = Cn[q];
        idx[q] = idxn[q];
      }
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kTile * kTile; i += blockDim.x) {
    const int ux = i % kTile, uy = i / kTile, j = TileIndex(ux, uy);
    const size_t o = (size_t(c) * 4 * N + ty0 + uy) * N + tx0 + ux;
#pragma unroll
    for (int k = 0; k < 4; ++k) lag[o + size_t(k) * N * N] = float2{sre[k][j], sim[k][j]};
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

// Bicubic weights of Keys' cubic convolution (a = -0.75) at offset f, in
// double precision without contraction, as skymon_taps computes them
__device__ inline void Keys4(double f, double* wt) {
  const double a = -0.75;
  const double t[4] = {__dadd_rn(1.0, f), f, __dsub_rn(1.0, f), __dsub_rn(2.0, f)};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const double t2 = __dmul_rn(t[k], t[k]), t3 = __dmul_rn(t2, t[k]);
    const double nr = __dadd_rn(__dsub_rn(__dmul_rn(a + 2, t3), __dmul_rn(a + 3, t2)), 1.0);
    const double fr = __dsub_rn(__dadd_rn(__dsub_rn(__dmul_rn(a, t3), __dmul_rn(5 * a, t2)), __dmul_rn(8 * a, t[k])),
                                4 * a);
    wt[k] = t[k] <= 1.0 ? nr : fr;
  }
}

// Stokes I of the monitor's pixel slots: out[c'][p] = sum of the 16 bicubic
// taps of slot p of sum_{k < nbin} (XX + YY)[c' nbin + k]. The positions are
// 1-based FITS pixels (x, y) of the image in FITS order; FITS pixel
// (i + 1, j + 1) is buffer element [a][b], a = (i + N/2) % N,
// b = (N - 1 - j + N/2) % N (save_output; sky_publisher.hpp).
__global__ void SkyRegrid(const float4* __restrict__ img, const double2* __restrict__ xy_a,
                          const double2* __restrict__ xy_b, double f, int P, int N, int nbin, int nout,
                          float* __restrict__ out) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= P) return;
  const double2 a2 = xy_a[p], b2 = xy_b[p];
  const double xx = __dadd_rn(a2.x, __dmul_rn(f, __dsub_rn(b2.x, a2.x)));
  const double yy = __dadd_rn(a2.y, __dmul_rn(f, __dsub_rn(b2.y, a2.y)));
  const double x = xx - 1.0, y = yy - 1.0;
  const double fx0 = floor(x), fy0 = floor(y);
  const bool ok = isfinite(fx0) && isfinite(fy0) && fx0 >= 1 && fy0 >= 1 && fx0 < N - 2 && fy0 < N - 2;
  if (!ok) {
    for (int c = 0; c < nout; ++c) out[size_t(c) * P + p] = 0.f;
    return;
  }
  const int x0 = int(fx0), y0 = int(fy0);
  double wx[4], wy[4];
  Keys4(x - x0, wx);
  Keys4(y - y0, wy);
  int cell[16];
  float w[16];
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int fi = x0 + i - 1, fj = y0 + j - 1;  // 0-based FITS pixel
      const int a = (fi + N / 2) % N, b = (N - 1 - fj + N / 2) % N;
      cell[4 * j + i] = a * N + b;
      w[4 * j + i] = float(__dmul_rn(wy[j], wx[i]));
    }
  const size_t nn = size_t(N) * N;
  for (int c = 0; c < nout; ++c) {
    float s = 0.f;
    for (int k = 0; k < nbin; ++k) {
      const float4* im = img + size_t(c * nbin + k) * nn;
      float v = 0.f;
#pragma unroll
      for (int t = 0; t < 16; ++t) {
        const float4 q = im[cell[t]];
        v += w[t] * (q.x + q.y);
      }
      s += v;
    }
    out[size_t(c) * P + p] = s;
  }
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
  // tiled gridding: the pairs touching every (channel, tile)
  bool tiled{true};
  uint32_t* d_list{};
  // the pairs' kernel cross-correlations [chan][bl][(2S - 1)^2] (float2, or
  // __half2 with half_ctab) over the per-channel scale d_cscale
  void* d_ctab{};
  float* d_cscale{};
  bool half_ctab{false};
  int* d_off{};
  float4* d_img[2]{};  // the images being combined and copied out, alternately
  cufftHandle plan{};
  cudaStream_t s{}, s_copy{};
  cudaEvent_t ev_free{}, ev_combined{}, ev_copied[2]{};
  int next{0};  // device image the next combination writes
  int last{0};  // device image of the last Image call
  // the all-sky monitor's pixel table
  double2* d_xy_a{};
  double2* d_xy_b{};
  float* d_pix[2]{};
  int sky_p{0}, sky_cap{0}, sky_nbin{1};
  bool sky_done{false};  // the last image was regridded
  double sky_ta{0}, sky_tb{1};
  // timing of the last image (gridding to combination, and the copy)
  cudaEvent_t t_grid{}, t_comb{}, t_copy{};
  bool timed{false};
  std::vector<float> img_ms, copy_ms;
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
    if (const char* e = std::getenv("EPIC_FX_GRIDDER")) tiled = std::string(e) != "atomic";
    if (N % kTile) tiled = false;
    // as much shared memory as the SM allows, for more resident tiles
    cudaFuncSetAttribute(GridLagsTiled<kSupport, float2>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(GridLagsTiled<kSupport, __half2>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    if (const char* e = std::getenv("EPIC_FX_CTAB")) half_ctab = std::string(e) == "half";
    for (auto& p : d_img) Check(cudaMalloc(&p, size_t(nchan) * N * N * sizeof(float4)), "image");
    int dims[2] = {N, N};
    CheckFft(cufftPlanMany(&plan, 2, dims, nullptr, 1, N * N, nullptr, 1, N * N, CUFFT_C2C, nchan * 4), "plan");
    Check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    Check(cudaStreamCreateWithFlags(&s_copy, cudaStreamNonBlocking), "stream");
    CheckFft(cufftSetStream(plan, s), "stream");
    for (auto* e : {&ev_free, &ev_combined, &ev_copied[0], &ev_copied[1]})
      Check(cudaEventCreateWithFlags(e, cudaEventDisableTiming), "event");
    for (auto* e : {&t_grid, &t_comb, &t_copy}) Check(cudaEventCreate(e), "event");
    LOG(INFO) << "FxImager: " << N << " x " << N << " images of " << nchan << " channels from the Tensor-Core "
              << "Correlator (" << nseq << " sequences per gulp), auto-correlations "
              << (keep_auto ? "kept" : "removed") << "; "
              << (size_t(nchan) * (kNbl * 4 * 8 + 4 * N * N * 8 + nseq_pad * kInputs * 2) >> 20) << " MiB on GPU "
              << device;
  }
  ~Impl() {
    cudaSetDevice(device);
    cudaStreamSynchronize(s);
    cudaStreamSynchronize(s_copy);
    cufftDestroy(plan);
    for (void* p : {static_cast<void*>(d_tcc_in), static_cast<void*>(d_vis), static_cast<void*>(d_base),
                    static_cast<void*>(d_K), static_cast<void*>(d_corr), static_cast<void*>(d_lag),
                    static_cast<void*>(d_img[0]), static_cast<void*>(d_img[1]), static_cast<void*>(d_list), d_ctab, static_cast<void*>(d_cscale),
                    static_cast<void*>(d_off), static_cast<void*>(d_xy_a),
                    static_cast<void*>(d_xy_b), static_cast<void*>(d_pix[0]), static_cast<void*>(d_pix[1])})
      if (p) cudaFree(p);
    for (auto e : {ev_free, ev_combined, ev_copied[0], ev_copied[1], t_grid, t_comb, t_copy}) cudaEventDestroy(e);
    cudaStreamDestroy(s);
    cudaStreamDestroy(s_copy);
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
  if (m->tiled) {
    const int s2 = 2 * m->S - 1;
    if (!m->d_ctab) {
      const size_t el = m->half_ctab ? sizeof(__half2) : sizeof(float2);
      Check(cudaMalloc(&m->d_ctab, size_t(m->nchan) * kNbl * s2 * s2 * el), "pair kernels");
      Check(cudaMalloc(&m->d_cscale, m->nchan * sizeof(float)), "pair kernels");
    }
    KernelScale<kSupport><<<m->nchan, kStands, 0, m->s>>>(m->d_K, m->d_cscale);
    if (m->half_ctab)
      PairKernels<kSupport><<<dim3(kNbl, m->nchan), 96, 0, m->s>>>(m->d_K, m->d_cscale,
                                                                  static_cast<__half2*>(m->d_ctab));
    else
      PairKernels<kSupport><<<dim3(kNbl, m->nchan), 96, 0, m->s>>>(m->d_K, m->d_cscale,
                                                                  static_cast<float2*>(m->d_ctab));
    Check(cudaGetLastError(), "pair kernels");
    // the pairs whose footprint touches each tile of each channel's lag grid
    std::vector<int2> base(static_cast<size_t>(ns));
    Check(cudaMemcpy(base.data(), m->d_base, base.size() * sizeof(int2), cudaMemcpyDeviceToHost), "cells");
    const int N = m->N, ntx = N / kTile, nt = ntx * ntx, r = m->S - 1;
    std::vector<uint32_t> list;
    list.reserve(size_t(m->nchan) * kNbl * 2);
    std::vector<std::vector<uint32_t>> per(nt);
    std::vector<int> off(size_t(m->nchan) * nt + 1, 0);
    for (int c = 0; c < m->nchan; ++c) {
      for (auto& v : per) v.clear();
      for (int b = 0; b < kStands; ++b)
        for (int a = 0; a <= b; ++a) {
          if (a == b && !m->keep_auto) continue;
          const int2 pa = base[c * kStands + a], pb = base[c * kStands + b];
          const int dx = pa.x - pb.x, dy = pa.y - pb.y;
          int tiles[4], nt4 = 0;
          for (int ey : {dy - r, dy + r})
            for (int ex : {dx - r, dx + r}) {
              const int t = (((ey % N) + N) % N / kTile) * ntx + (((ex % N) + N) % N / kTile);
              bool seen = false;
              for (int q = 0; q < nt4; ++q) seen |= tiles[q] == t;
              if (!seen) tiles[nt4++] = t;
            }
          const uint32_t ent = uint32_t(b * (b + 1) / 2 + a) | (uint32_t(a) << 16) | (uint32_t(b) << 24);
          for (int q = 0; q < nt4; ++q) per[tiles[q]].push_back(ent);
        }
      for (int t = 0; t < nt; ++t) {
        off[size_t(c) * nt + t] = int(list.size());
        list.insert(list.end(), per[t].begin(), per[t].end());
      }
    }
    off[size_t(m->nchan) * nt] = int(list.size());
    if (m->d_list) cudaFree(m->d_list);
    if (m->d_off) cudaFree(m->d_off);
    Check(cudaMalloc(&m->d_list, list.size() * sizeof(uint32_t)), "pair lists");
    Check(cudaMalloc(&m->d_off, off.size() * sizeof(int)), "pair lists");
    Check(cudaMemcpy(m->d_list, list.data(), list.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "lists");
    Check(cudaMemcpy(m->d_off, off.data(), off.size() * sizeof(int), cudaMemcpyHostToDevice), "lists");
    LOG(INFO) << "FxImager: tiled gridding, " << list.size() << " pair entries in " << nt << " tiles x "
              << m->nchan << " channels";
  }
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

void FxImager::SetSkyTable(const double* p_xy_a, const double* p_xy_b, int p_capacity, double p_t_a, double p_t_b,
                           int p_nbin) {
  Check(cudaSetDevice(m->device), "device");
  if (p_nbin < 1 || m->nchan % p_nbin) throw std::invalid_argument("FxImager: channels not divisible by sky_nbin");
  // the copies of earlier images may still read the pixel buffers
  Check(cudaStreamSynchronize(m->s), "sky table");
  Check(cudaStreamSynchronize(m->s_copy), "sky table");
  if (p_capacity > m->sky_cap) {
    for (void* p : {static_cast<void*>(m->d_xy_a), static_cast<void*>(m->d_xy_b), static_cast<void*>(m->d_pix[0]),
                    static_cast<void*>(m->d_pix[1])})
      if (p) cudaFree(p);
    Check(cudaMalloc(&m->d_xy_a, size_t(p_capacity) * sizeof(double2)), "sky table");
    Check(cudaMalloc(&m->d_xy_b, size_t(p_capacity) * sizeof(double2)), "sky table");
    for (auto& p : m->d_pix) Check(cudaMalloc(&p, size_t(m->nchan) * p_capacity * sizeof(float)), "sky pixels");
    m->sky_cap = p_capacity;
  }
  Check(cudaMemcpy(m->d_xy_a, p_xy_a, size_t(p_capacity) * sizeof(double2), cudaMemcpyHostToDevice), "sky table");
  Check(cudaMemcpy(m->d_xy_b, p_xy_b, size_t(p_capacity) * sizeof(double2), cudaMemcpyHostToDevice), "sky table");
  m->sky_p = p_capacity;
  m->sky_ta = p_t_a;
  m->sky_tb = p_t_b;
  m->sky_nbin = p_nbin;
}

int FxImager::SkyPixels() const { return m->sky_p; }
bool FxImager::SkyDone() const { return m->sky_done; }

void FxImager::Image(const float* p_phases, float* p_host, double p_unix, float* p_pix_host) {
  if (!m->geometry) throw std::logic_error("FxImager: Image before SetGeometry");
  const int N = m->N;
  const size_t nlag = size_t(m->nchan) * 4 * N * N;
  Check(cudaEventRecord(m->t_grid, m->s), "event");
  if (m->tiled) {
    const int nt = (N / kTile) * (N / kTile);
    const auto* g = reinterpret_cast<const float4*>(p_phases);
    if (m->half_ctab)
      GridLagsTiled<kSupport><<<dim3(nt, m->nchan), kTileThreads, 0, m->s>>>(
          m->d_vis, m->d_base, static_cast<const __half2*>(m->d_ctab), m->d_cscale, g, m->d_list, m->d_off, m->d_lag,
          N);
    else
      GridLagsTiled<kSupport><<<dim3(nt, m->nchan), kTileThreads, 0, m->s>>>(
          m->d_vis, m->d_base, static_cast<const float2*>(m->d_ctab), m->d_cscale, g, m->d_list, m->d_off, m->d_lag,
          N);
  } else {
    Check(cudaMemsetAsync(m->d_lag, 0, nlag * sizeof(float2), m->s), "clear");
    const dim3 blk((2 * kSupport - 1) * (2 * kSupport - 1), kPairs), grd((kNbl + kPairs - 1) / kPairs, m->nchan);
    GridLags<kSupport><<<grd, blk, 0, m->s>>>(m->d_vis, m->d_base, m->d_K, reinterpret_cast<const float4*>(p_phases),
                                              m->d_lag, N, m->keep_auto);
  }
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
  const int k = m->next;
  m->next ^= 1;
  // the copy that last read this device image must be done before it is overwritten
  Check(cudaStreamWaitEvent(m->s, m->ev_copied[k], 0), "wait copy");
  CombineImages<<<dim3((N * N + 255) / 256, m->nchan), 256, 0, m->s>>>(m->d_lag, m->d_img[k], m->d_corr, N,
                                                                     r * r * r * r / 32.f);
  Check(cudaGetLastError(), "combine");
  // within the table's span (with the monitor's margin): a table the monitor
  // stopped updating is not used, and the image is published instead
  const double fr = (p_unix - m->sky_ta) / (m->sky_tb - m->sky_ta);
  const bool sky = m->sky_p > 0 && p_pix_host && fr >= -0.5 && fr <= 1.5;
  if (m->sky_p > 0 && sky != m->sky_done) {
    LOG(INFO) << "FxImager: " << (sky ? "regridding to" : "not regridding to") << " the monitor's pixels (image at "
              << std::fixed << p_unix << ", table " << m->sky_ta << " - " << m->sky_tb << ")";
  }
  m->sky_done = sky;
  if (sky) {
    // the monitor's interpolation between its two positions (geometry.SkyGrid.taps)
    const double f = std::min(std::max(fr, 0.0), 1.5);
    SkyRegrid<<<(m->sky_p + 127) / 128, 128, 0, m->s>>>(m->d_img[k], m->d_xy_a, m->d_xy_b, f, m->sky_p, N,
                                                        m->sky_nbin, m->nchan / m->sky_nbin, m->d_pix[k]);
    Check(cudaGetLastError(), "sky regrid");
  }
  Check(cudaEventRecord(m->ev_combined, m->s), "event");
  Check(cudaEventRecord(m->t_comb, m->s), "event");
  Check(cudaStreamWaitEvent(m->s_copy, m->ev_combined, 0), "wait combine");
  if (p_host) {
    Check(cudaMemcpyAsync(p_host, m->d_img[k], size_t(m->nchan) * N * N * 4 * sizeof(float),
                          cudaMemcpyDeviceToHost, m->s_copy),
          "copy");
  }
  if (sky) {
    Check(cudaMemcpyAsync(p_pix_host, m->d_pix[k], size_t(m->nchan / m->sky_nbin) * m->sky_p * sizeof(float),
                          cudaMemcpyDeviceToHost, m->s_copy),
          "copy");
  }
  Check(cudaEventRecord(m->ev_copied[k], m->s_copy), "event");
  Check(cudaEventRecord(m->t_copy, m->s_copy), "event");
  m->last = k;
  m->timed = true;
}

void FxImager::Sync() {
  Check(cudaEventSynchronize(m->ev_copied[m->last]), "image");
  if (!m->timed) return;
  m->timed = false;
  float a = 0.f, b = 0.f;
  cudaEventElapsedTime(&a, m->t_grid, m->t_comb);
  cudaEventElapsedTime(&b, m->t_comb, m->t_copy);
  m->img_ms.push_back(a);
  m->copy_ms.push_back(b);
  if (m->img_ms.size() == 240) {
    auto stat = [](std::vector<float>& v, float* mean, float* p99) {
      std::sort(v.begin(), v.end());
      double s = 0;
      for (float x : v) s += x;
      *mean = float(s / v.size());
      *p99 = v[std::min(v.size() - 1, size_t(0.99 * v.size()))];
      v.clear();
    };
    float im, ip, cm, cp;
    stat(m->img_ms, &im, &ip);
    stat(m->copy_ms, &cm, &cp);
    LOG(INFO) << "FxImager: grid + transform + combine " << im << " ms (p99 " << ip << "), copy to host " << cm
              << " ms (p99 " << cp << ") per image, over 240 images";
  }
}
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
void FxImager::Image(const float*, float*, double, float*) {}
void FxImager::SetSkyTable(const double*, const double*, int, double, double, int) {}
int FxImager::SkyPixels() const { return 0; }
bool FxImager::SkyDone() const { return false; }
void FxImager::Sync() {}
cudaEvent_t FxImager::InputFree() const { return nullptr; }
cudaStream_t FxImager::Stream() const { return nullptr; }
int FxImager::N() const { return 0; }

#endif
