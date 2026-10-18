// Copyright (c) 2026 The EPIC++ authors
//
// Interference stage before imaging; see rfi_stage.h.
#include "./rfi_stage.h"

#include <glog/logging.h>

#include <algorithm>

namespace {
constexpr int kStands = LWA_SV_NSTANDS;
constexpr int kIn = LWA_SV_NSTANDS * LWA_SV_NPOLS;
constexpr int kFlagThreads = 1024;
constexpr int kMaxSortSeq = 4096;  // impulse detection needs the gulp's sequences in shared memory

void Check(cudaError_t p_err, const char* p_what) {
  LOG_IF(FATAL, p_err != cudaSuccess) << "RfiStage: " << p_what << ": " << cudaGetErrorString(p_err);
}

// Counter-based uniform deviate in [-0.5, 0.5) (lowbias32 hash)
__device__ __forceinline__ float Dither(uint32_t p_a, uint32_t p_b) {
  uint32_t h = p_a * 0x9E3779B1u ^ (p_b + 0x7F4A7C15u + (p_a << 6) + (p_a >> 2));
  h ^= h >> 16;
  h *= 0x7feb352du;
  h ^= h >> 15;
  h *= 0x846ca68bu;
  h ^= h >> 16;
  return float(h >> 8) * (1.0f / 16777216.0f) - 0.5f;
}

// Symmetric clip at +-7, as the F-engine data; -8 would bias the mean
__device__ __forceinline__ int Q4(float p_v) { return __float2int_rn(fminf(fmaxf(p_v, -7.f), 7.f)); }

__device__ __forceinline__ float2 Unpack(uint8_t p_b) {
  return make_float2(float(int8_t(p_b) >> 4), float(int8_t(uint8_t(p_b << 4)) >> 4));
}

__device__ __forceinline__ uint8_t Pack(int p_re, int p_im) { return uint8_t(((p_re & 0xF) << 4) | (p_im & 0xF)); }

// One block per (sequence, channel), one thread per stand (its X and Y inputs).
// Q = 0: copy and band power only.
template <int Q>
__global__ void RfiProject(const uint8_t* __restrict__ p_in, uint8_t* __restrict__ p_out,
                           const float2* __restrict__ p_ut, const float2* __restrict__ p_vt, int p_nseq,
                           float* __restrict__ p_power, uint32_t p_key) {
  const int t = blockIdx.x, c = blockIdx.y, s = threadIdx.x;
  const size_t base = (size_t(c) * p_nseq + t) * kIn;
  const uint16_t w = *reinterpret_cast<const uint16_t*>(p_in + base + 2 * s);
  uint16_t o = w;
  float pw;
  if constexpr (Q == 0) {
    const float2 x = Unpack(uint8_t(w & 0xFF)), y = Unpack(uint8_t(w >> 8));
    pw = x.x * x.x + x.y * x.y + y.x * y.x + y.y * y.y;
  } else {
    float2 x = Unpack(uint8_t(w & 0xFF)), y = Unpack(uint8_t(w >> 8));
    __shared__ float2 red[Q][2][kStands / 32];
    const float2* vc = p_vt + size_t(c) * kProjQmax * kIn;
    const float2* uc = p_ut + size_t(c) * kProjQmax * kIn;
    // s_k = Vt_k^H x, separately for the X and Y inputs (each component acts on one polarisation)
#pragma unroll
    for (int k = 0; k < Q; ++k) {
      const float2 vx = vc[k * kIn + 2 * s], vy = vc[k * kIn + 2 * s + 1];
      float2 ax = make_float2(vx.x * x.x + vx.y * x.y, vx.x * x.y - vx.y * x.x);
      float2 ay = make_float2(vy.x * y.x + vy.y * y.y, vy.x * y.y - vy.y * y.x);
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        ax.x += __shfl_xor_sync(0xffffffff, ax.x, off);
        ax.y += __shfl_xor_sync(0xffffffff, ax.y, off);
        ay.x += __shfl_xor_sync(0xffffffff, ay.x, off);
        ay.y += __shfl_xor_sync(0xffffffff, ay.y, off);
      }
      if ((s & 31) == 0) {
        red[k][0][s >> 5] = ax;
        red[k][1][s >> 5] = ay;
      }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < Q; ++k) {
      float2 sx = make_float2(0.f, 0.f), sy = make_float2(0.f, 0.f);
#pragma unroll
      for (int i = 0; i < kStands / 32; ++i) {
        sx.x += red[k][0][i].x;
        sx.y += red[k][0][i].y;
        sy.x += red[k][1][i].x;
        sy.y += red[k][1][i].y;
      }
      const float2 ux = uc[k * kIn + 2 * s], uy = uc[k * kIn + 2 * s + 1];
      x.x -= ux.x * sx.x - ux.y * sx.y;
      x.y -= ux.x * sx.y + ux.y * sx.x;
      y.x -= uy.x * sy.x - uy.y * sy.y;
      y.y -= uy.x * sy.y + uy.y * sy.x;
    }
    const uint32_t a = (p_key * 0x01000193u) ^ uint32_t(base);
    const int xr = Q4(x.x + Dither(a, 4 * s)), xi = Q4(x.y + Dither(a, 4 * s + 1));
    const int yr = Q4(y.x + Dither(a, 4 * s + 2)), yi = Q4(y.y + Dither(a, 4 * s + 3));
    o = uint16_t(Pack(xr, xi)) | uint16_t(uint16_t(Pack(yr, yi)) << 8);
    pw = float(xr * xr + xi * xi + yr * yr + yi * yi);
  }
  *reinterpret_cast<uint16_t*>(p_out + base + 2 * s) = o;
  // band power of this (channel, sequence)
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) pw += __shfl_xor_sync(0xffffffff, pw, off);
  __shared__ float pred[kStands / 32];
  if ((s & 31) == 0) pred[s >> 5] = pw;
  __syncthreads();
  if (s == 0) {
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < kStands / 32; ++i) sum += pred[i];
    p_power[size_t(c) * p_nseq + t] = sum;
  }
}

__device__ void BitonicSort(float* p_v, int p_n2) {
  for (int k = 2; k <= p_n2; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < p_n2; i += blockDim.x) {
        const int ixj = i ^ j;
        if (ixj > i) {
          const bool asc = (i & k) == 0;
          const float a = p_v[i], b = p_v[ixj];
          if ((a > b) == asc) {
            p_v[i] = b;
            p_v[ixj] = a;
          }
        }
      }
      __syncthreads();
    }
  }
}

// One block per stream: band power over the stream's channels per sequence,
// robust threshold, zero the sequences above it.
__global__ void ImpulseFlag(const float* __restrict__ p_power, int p_nchan, int p_nseq, int p_n2, float p_sigma,
                            uint8_t* __restrict__ p_out, uint32_t* __restrict__ p_count) {
  extern __shared__ float sh[];
  float* band = sh;        // [p_n2]
  float* work = sh + p_n2;  // [p_n2]
  for (int t = threadIdx.x; t < p_n2; t += blockDim.x) {
    float b = 0.f;
    if (t < p_nseq) {
      for (int c = 0; c < p_nchan; ++c) b += p_power[size_t(c) * p_nseq + t];
    }
    band[t] = b;
    work[t] = t < p_nseq ? b : INFINITY;
  }
  __syncthreads();
  BitonicSort(work, p_n2);
  const float med = work[p_nseq / 2];
  __syncthreads();
  for (int t = threadIdx.x; t < p_n2; t += blockDim.x) work[t] = t < p_nseq ? fabsf(band[t] - med) : INFINITY;
  __syncthreads();
  BitonicSort(work, p_n2);
  const float sig = 1.4826f * work[p_nseq / 2];
  if (!(sig > 0.f)) return;  // empty or constant gulp
  const float thr = med + p_sigma * sig;
  uint32_t n = 0;
  for (int t = threadIdx.x; t < p_nseq; t += blockDim.x) {
    if (band[t] > thr) {
      ++n;
      for (int c = 0; c < p_nchan; ++c) {
        uint4* d = reinterpret_cast<uint4*>(p_out + (size_t(c) * p_nseq + t) * kIn);
#pragma unroll
        for (int i = 0; i < kIn / 16; ++i) d[i] = make_uint4(0, 0, 0, 0);
      }
    }
  }
  if (n) atomicAdd(p_count, n);
}
}  // namespace

RfiStage::RfiStage(int p_device, float p_impulse_sigma) : m_device(p_device), m_sigma(p_impulse_sigma) {
  Check(cudaSetDevice(m_device), "cudaSetDevice");
  Check(cudaMalloc(&m_cnt_d, MAX_GULP_STREAMS * sizeof(uint32_t)), "cudaMalloc counters");
  Check(cudaMemset(m_cnt_d, 0, MAX_GULP_STREAMS * sizeof(uint32_t)), "cudaMemset counters");
  Check(cudaMallocHost(&m_cnt_h, MAX_GULP_STREAMS * sizeof(uint32_t)), "cudaMallocHost counters");
  std::fill(m_cnt_h, m_cnt_h + MAX_GULP_STREAMS, 0u);
}

RfiStage::~RfiStage() {
  cudaSetDevice(m_device);
  cudaFree(m_out);
  cudaFree(m_ut);
  cudaFree(m_vt);
  cudaFree(m_power);
  cudaFree(m_cnt_d);
  cudaFreeHost(m_cnt_h);
}

void RfiStage::Resize(int p_nchan, int p_nseq) {
  Check(cudaSetDevice(m_device), "cudaSetDevice");
  LOG_IF(FATAL, m_sigma > 0.f && p_nseq > kMaxSortSeq)
      << "RfiStage: impulse detection supports at most " << kMaxSortSeq << " sequences per gulp";
  cudaFree(m_out);
  cudaFree(m_ut);
  cudaFree(m_vt);
  cudaFree(m_power);
  m_nchan = p_nchan;
  m_nseq = p_nseq;
  Check(cudaMalloc(&m_out, size_t(p_nchan) * p_nseq * kIn), "cudaMalloc gulp");
  Check(cudaMalloc(&m_ut, size_t(p_nchan) * kProjQmax * kIn * sizeof(float2)), "cudaMalloc projector");
  Check(cudaMalloc(&m_vt, size_t(p_nchan) * kProjQmax * kIn * sizeof(float2)), "cudaMalloc projector");
  Check(cudaMalloc(&m_power, size_t(p_nchan) * p_nseq * sizeof(float)), "cudaMalloc power");
  m_q = 0;
  LOG(INFO) << "RfiStage: " << p_nchan << " channels x " << p_nseq << " sequences, projector up to " << kProjQmax
            << " components per polarisation"
            << (m_sigma > 0.f ? ", impulse detection at " + std::to_string(m_sigma) + " sigma" : "");
}

void RfiStage::SetProjector(const std::complex<float>* p_ut, const std::complex<float>* p_vt, int p_q,
                            int p_nchan) {
  Check(cudaSetDevice(m_device), "cudaSetDevice");
  LOG_IF(FATAL, p_nchan != m_nchan) << "RfiStage: projector for " << p_nchan << " channels, gulps have " << m_nchan;
  const size_t n = size_t(p_nchan) * kProjQmax * kIn * sizeof(float2);
  Check(cudaMemcpy(m_ut, p_ut, n, cudaMemcpyHostToDevice), "upload Ut");
  Check(cudaMemcpy(m_vt, p_vt, n, cudaMemcpyHostToDevice), "upload Vt");
  m_q = std::max(0, std::min(p_q, kProjQmax));
}

void RfiStage::Run(const uint8_t* p_gulp, int p_chan_offset, int p_nchan, cudaStream_t p_stream, int p_si) {
  const size_t off = size_t(p_chan_offset) * m_nseq * kIn;
  const dim3 grid(m_nseq, p_nchan);
  const float2* ut = m_ut + size_t(p_chan_offset) * kProjQmax * kIn;
  const float2* vt = m_vt + size_t(p_chan_offset) * kProjQmax * kIn;
  float* pw = m_power + size_t(p_chan_offset) * m_nseq;
  const uint32_t key = m_gulp * 0x9E3779B9u ^ uint32_t(p_chan_offset) * 0x85EBCA6Bu;
  switch (m_q) {
    case 0: RfiProject<0><<<grid, kStands, 0, p_stream>>>(p_gulp + off, m_out + off, ut, vt, m_nseq, pw, key); break;
    case 1: RfiProject<1><<<grid, kStands, 0, p_stream>>>(p_gulp + off, m_out + off, ut, vt, m_nseq, pw, key); break;
    case 2: RfiProject<2><<<grid, kStands, 0, p_stream>>>(p_gulp + off, m_out + off, ut, vt, m_nseq, pw, key); break;
    case 3: RfiProject<3><<<grid, kStands, 0, p_stream>>>(p_gulp + off, m_out + off, ut, vt, m_nseq, pw, key); break;
    default: RfiProject<4><<<grid, kStands, 0, p_stream>>>(p_gulp + off, m_out + off, ut, vt, m_nseq, pw, key); break;
  }
  Check(cudaGetLastError(), "RfiProject launch");
  if (m_sigma > 0.f) {
    int n2 = 1;
    while (n2 < m_nseq) n2 <<= 1;
    ImpulseFlag<<<1, kFlagThreads, 2 * n2 * sizeof(float), p_stream>>>(pw, p_nchan, m_nseq, n2, m_sigma, m_out + off,
                                                                        m_cnt_d + p_si);
    Check(cudaGetLastError(), "ImpulseFlag launch");
  }
}

void RfiStage::EndImage(int p_si, cudaStream_t p_stream) {
  Check(cudaMemcpyAsync(m_cnt_h + p_si, m_cnt_d + p_si, sizeof(uint32_t), cudaMemcpyDeviceToHost, p_stream),
        "count read-back");
  Check(cudaMemsetAsync(m_cnt_d + p_si, 0, sizeof(uint32_t), p_stream), "count reset");
}

uint64_t RfiStage::CollectFlagged() {
  uint64_t n = 0;
  for (int i = 0; i < MAX_GULP_STREAMS; ++i) {
    n += m_cnt_h[i];
    m_cnt_h[i] = 0;
  }
  return n;
}
