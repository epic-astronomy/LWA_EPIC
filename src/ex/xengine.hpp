// Copyright (c) 2026 The EPIC++ authors
//
// CPU X-engine for the calibration tap (see cal_tap.hpp).
//
// Input: an epic++ gulp in CHAN_MAJOR order, [chan][seq][stand 256][pol 2]
// bytes, each byte a CNib {signed char im:4, re:4} (im = low nibble).
// Output: per channel the XX and YY visibility matrices (upper triangle incl.
// autos), R_ab = sum_t x_a(t) conj(x_b(t)), accumulated in int64.
//
// Each channel is unpacked to int16 (re, im) pairs, time contiguous per input,
// plus a swizzled copy (-im, re). Then Re R_ab = madd(a, b) and
// Im R_ab = madd(a, bs) over 8 samples per 256-bit register (vpmaddwd), in
// 2x3 register tiles. Measured 13 GCMAC/s per Zen 3 core on epic1, so one core
// correlates ~7% of the samples of all 128 channels (XX and YY).
#ifndef SRC_EX_XENGINE_HPP_
#define SRC_EX_XENGINE_HPP_
#include <immintrin.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace xengine {
constexpr int kStands = 256;
constexpr int kPols = 2;
constexpr int kInputs = kStands * kPols;

inline int16_t lo_nib(uint8_t v) { return int16_t(int8_t(uint8_t(v << 4)) >> 4); }
inline int16_t hi_nib(uint8_t v) { return int16_t(int8_t(v) >> 4); }

// Unpack one channel, nseq samples (padded to a multiple of 8 with zeros),
// to a[pol][stand][t] (re, im) and as[pol][stand][t] (-im, re) int16 pairs.
inline void unpack_channel(const uint8_t* chan_base, int nseq, int nseq_pad, int16_t* a, int16_t* as) {
  for (int p = 0; p < kPols; ++p) {
    for (int s = 0; s < kStands; ++s) {
      int16_t* da = a + (size_t(p) * kStands + s) * nseq_pad * 2;
      int16_t* ds = as + (size_t(p) * kStands + s) * nseq_pad * 2;
      for (int t = 0; t < nseq; ++t) {
        uint8_t v = chan_base[size_t(t) * kInputs + s * kPols + p];
        int16_t re = hi_nib(v), im = lo_nib(v);
        da[2 * t] = re;
        da[2 * t + 1] = im;
        ds[2 * t] = int16_t(-im);
        ds[2 * t + 1] = re;
      }
      for (int t = nseq; t < nseq_pad; ++t) {
        da[2 * t] = da[2 * t + 1] = ds[2 * t] = ds[2 * t + 1] = 0;
      }
    }
  }
}

inline int32_t hsum(__m256i v) {
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
  return _mm_cvtsi128_si32(s);
}

// Correlate one polarisation: in a[stand][t], as[stand][t]; out re/im [256][256] (upper triangle).
inline void correlate_pol(const int16_t* a, const int16_t* as, int nseq_pad, int64_t* out_re, int64_t* out_im) {
  const int nvec = nseq_pad / 8;  // 8 complex samples per __m256i
  const size_t stride = size_t(nseq_pad) * 2;
  for (int i0 = 0; i0 < kStands; i0 += 2) {
    for (int j0 = i0; j0 < kStands; j0 += 3) {
      __m256i acc_re[2][3], acc_im[2][3];
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j) acc_re[i][j] = acc_im[i][j] = _mm256_setzero_si256();
      const int jn = (j0 + 3 <= kStands) ? 3 : kStands - j0;
      const __m256i* A0 = reinterpret_cast<const __m256i*>(a + size_t(i0) * stride);
      const __m256i* A1 = reinterpret_cast<const __m256i*>(a + size_t(i0 + 1) * stride);
      for (int v = 0; v < nvec; ++v) {
        __m256i x0 = _mm256_load_si256(A0 + v);
        __m256i x1 = _mm256_load_si256(A1 + v);
        for (int j = 0; j < jn; ++j) {
          const __m256i* B = reinterpret_cast<const __m256i*>(a + size_t(j0 + j) * stride);
          const __m256i* Bs = reinterpret_cast<const __m256i*>(as + size_t(j0 + j) * stride);
          __m256i y = _mm256_load_si256(B + v);
          __m256i ys = _mm256_load_si256(Bs + v);
          acc_re[0][j] = _mm256_add_epi32(acc_re[0][j], _mm256_madd_epi16(x0, y));
          acc_im[0][j] = _mm256_add_epi32(acc_im[0][j], _mm256_madd_epi16(x0, ys));
          acc_re[1][j] = _mm256_add_epi32(acc_re[1][j], _mm256_madd_epi16(x1, y));
          acc_im[1][j] = _mm256_add_epi32(acc_im[1][j], _mm256_madd_epi16(x1, ys));
        }
      }
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < jn; ++j) {
          int ia = i0 + i, jb = j0 + j;
          if (jb < ia) continue;
          out_re[ia * kStands + jb] += hsum(acc_re[i][j]);
          out_im[ia * kStands + jb] += hsum(acc_im[i][j]);
        }
    }
  }
}

// Accumulate nchan channels from a gulp. seq0/nseq select the tapped samples
// within the gulp (gulp holds gulp_nseq samples per channel). out_re/out_im are
// [nchan][2 pols][256][256] int64, accumulated in place.
inline void accumulate(const uint8_t* gulp, int gulp_nseq, int chan0, int nchan, int seq0, int nseq,
                        int64_t* out_re, int64_t* out_im, int nthreads = 1) {
  const int nseq_pad = (nseq + 7) / 8 * 8;
#pragma omp parallel num_threads(nthreads)
  {
    const size_t per = size_t(kPols) * kStands * nseq_pad * 2;
    int16_t* a = static_cast<int16_t*>(aligned_alloc(64, per * sizeof(int16_t)));
    int16_t* as = static_cast<int16_t*>(aligned_alloc(64, per * sizeof(int16_t)));
#pragma omp for schedule(dynamic, 1)
    for (int c = 0; c < nchan; ++c) {
      const uint8_t* base = gulp + (size_t(chan0 + c) * gulp_nseq + seq0) * kInputs;
      unpack_channel(base, nseq, nseq_pad, a, as);
      for (int p = 0; p < kPols; ++p) {
        size_t o = (size_t(c) * kPols + p) * kStands * kStands;
        correlate_pol(a + size_t(p) * kStands * nseq_pad * 2, as + size_t(p) * kStands * nseq_pad * 2,
                      nseq_pad, out_re + o, out_im + o);
      }
    }
    free(a);
    free(as);
  }
}

}  // namespace xengine
#endif  // SRC_EX_XENGINE_HPP_
