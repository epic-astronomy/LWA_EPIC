/*
 * Compute kernels of the EPIC all-sky monitor (skymon).
 *
 * Built by skymon/kernels.py with gcc -O3 -fopenmp and called through ctypes,
 * which releases the GIL for the duration of each call.
 *
 * Array layouts keep the pixel axis last: [chan][time][pix] and
 * [delay][time][pix]. Each OpenMP thread transforms a tile of pixels in its own
 * buffers, so all inner loops run over contiguous memory.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* ---------------------------------------------------------------- plan --- */
/*
 * plan layout (int32), produced by kernels.py from dedisp.fdmt_plan:
 *   [C] init_nd
 *   for each merge level, for each pair in band order:
 *     nd, d_lo[nd + 1], d_up[nd + 1]
 */

static int max_rows(const int* plan, int C, int* final_rows) {
  int rows = 0;
  for (int c = 0; c < C; ++c) rows += plan[c] + 1;
  int best = rows;
  const int* q = plan + C;
  for (int n = C / 2; n >= 1; n /= 2) {
    rows = 0;
    for (int k = 0; k < n; ++k) {
      int nd = q[0];
      rows += nd + 1;
      q += 1 + 2 * (nd + 1);
    }
    if (rows > best) best = rows;
    if (n == 1) *final_rows = rows;
  }
  return best;
}

/* FDMT of one tile: in [C][W][Pt] -> rows in buf, returns pointer to the final rows */
static float* fdmt_tile(const float* in, int C, int W, int Pt, const int* plan,
                        float* bufa, float* bufb, int* offs, int* nds) {
  const size_t slab = (size_t)W * Pt;
  /* initialisation: inside channel c the track spans d + 1 samples */
  int off = 0;
  for (int c = 0; c < C; ++c) {
    int nd = plan[c];
    offs[c] = off;
    nds[c] = nd;
    const float* x = in + (size_t)c * slab;
    float* r0 = bufa + (size_t)off * slab;
    memcpy(r0, x, slab * sizeof(float));
    for (int d = 1; d <= nd; ++d) {
      float* r = bufa + (size_t)(off + d) * slab;
      const float* rp = r - slab;
      memcpy(r, rp, (size_t)d * Pt * sizeof(float));
      const size_t n = (size_t)(W - d) * Pt;
      float* rd = r + (size_t)d * Pt;
      const float* rpd = rp + (size_t)d * Pt;
      for (size_t i = 0; i < n; ++i) rd[i] = rpd[i] + x[i];
    }
    off += nd + 1;
  }
  /* merge adjacent subbands; the lower one is reached last */
  const int* q = plan + C;
  float* src = bufa;
  float* dst = bufb;
  int nsub = C;
  while (nsub > 1) {
    int off_new = 0;
    for (int k = 0; k < nsub / 2; ++k) {
      int nd = q[0];
      const int* d_lo = q + 1;
      const int* d_up = q + 1 + (nd + 1);
      const float* alo = src + (size_t)offs[2 * k] * slab;
      const float* aup = src + (size_t)offs[2 * k + 1] * slab;
      for (int d = 0; d <= nd; ++d) {
        int lo = d_lo[d], up = d_up[d];
        const float* l = alo + (size_t)lo * slab;
        const float* u = aup + (size_t)up * slab;
        float* o = dst + (size_t)(off_new + d) * slab;
        const size_t head = (size_t)lo * Pt;
        memcpy(o, l, head * sizeof(float));
        const size_t n = slab - head;
        const float* lh = l + head;
        float* oh = o + head;
        for (size_t i = 0; i < n; ++i) oh[i] = lh[i] + u[i];
      }
      /* record the merged layout in the slot of the pair's first member */
      offs[k] = off_new;
      nds[k] = nd;
      off_new += nd + 1;
      q += 1 + 2 * (nd + 1);
    }
    nsub /= 2;
    float* t = src;
    src = dst;
    dst = t;
  }
  return src; /* rows 0..D at offset 0 */
}

static void load_tile(const float* data, int C, int W, int P, int p0, int Pt, float* tile) {
  for (int c = 0; c < C; ++c)
    for (int t = 0; t < W; ++t) {
      const float* s = data + ((size_t)c * W + t) * P + p0;
      float* d = tile + ((size_t)c * W + t) * Pt;
      memcpy(d, s, (size_t)Pt * sizeof(float));
    }
}

/*
 * Remove a straight line, fitted by least squares over the window, from each
 * channel of each pixel in a tile [C][W][Pt]. A source rising through the
 * primary beam or a slow gain change is a ramp that the running statistics
 * follow only with a lag; across all channels such a ramp would add up like a
 * broadband signal. A burst fills about one sample per channel, so the fit
 * takes away about 1/W of it.
 */
static void detrend_tile(float* tile, int C, int W, int Pt, double* sx, double* stx) {
  const double tm = 0.5 * (W - 1);
  double stt = 0;
  for (int t = 0; t < W; ++t) stt += (t - tm) * (t - tm);
  for (int c = 0; c < C; ++c) {
    float* x = tile + (size_t)c * W * Pt;
    for (int p = 0; p < Pt; ++p) { sx[p] = 0; stx[p] = 0; }
    for (int t = 0; t < W; ++t) {
      const float* r = x + (size_t)t * Pt;
      const double u = t - tm;
      for (int p = 0; p < Pt; ++p) { sx[p] += r[p]; stx[p] += u * r[p]; }
    }
    for (int p = 0; p < Pt; ++p) { sx[p] /= W; stx[p] /= stt; }
    for (int t = 0; t < W; ++t) {
      float* r = x + (size_t)t * Pt;
      const double u = t - tm;
      for (int p = 0; p < Pt; ++p) r[p] -= (float)(sx[p] + u * stx[p]);
    }
  }
}

/* Remove the mean over the window from each channel of each pixel of a tile */
static void demean_tile(float* tile, int C, int W, int Pt, double* sx) {
  for (int c = 0; c < C; ++c) {
    float* x = tile + (size_t)c * W * Pt;
    for (int p = 0; p < Pt; ++p) sx[p] = 0;
    for (int t = 0; t < W; ++t) {
      const float* r = x + (size_t)t * Pt;
      for (int p = 0; p < Pt; ++p) sx[p] += r[p];
    }
    for (int p = 0; p < Pt; ++p) sx[p] /= W;
    for (int t = 0; t < W; ++t) {
      float* r = x + (size_t)t * Pt;
      for (int p = 0; p < Pt; ++p) r[p] -= (float)sx[p];
    }
  }
}

/*
 * Zero-DM filter (Eatough et al. 2009): subtract from every sample of a pixel
 * the mean over channels at that time. It removes whatever changes in all
 * channels at once, such as gain changes or the slow ripple of interpolating
 * a bright source. A burst above half the tier's delay range lights one or
 * two channels per sample, so it loses one or two parts in nchan.
 */
static void zerodm_tile(float* tile, int C, int W, int Pt) {
  const float inv = 1.f / C;
  float* m = malloc(sizeof(float) * (size_t)Pt);
  for (int t = 0; t < W; ++t) {
    for (int p = 0; p < Pt; ++p) m[p] = 0.f;
    for (int c = 0; c < C; ++c) {
      const float* r = tile + ((size_t)c * W + t) * Pt;
      for (int p = 0; p < Pt; ++p) m[p] += r[p];
    }
    for (int c = 0; c < C; ++c) {
      float* r = tile + ((size_t)c * W + t) * Pt;
      for (int p = 0; p < Pt; ++p) r[p] -= m[p] * inv;
    }
  }
  free(m);
}

/* Full FDMT, out [D + 1][W][P]. Used by the tests and for candidate products. */
int skymon_fdmt(const float* data, int C, int W, int P, const int* plan, float* out,
                int nthreads, int ptile) {
  int final_rows = 0;
  const int rows = max_rows(plan, C, &final_rows);
  if (ptile <= 0) ptile = 16;
  const int ntile = (P + ptile - 1) / ptile;
#pragma omp parallel num_threads(nthreads)
  {
    float* tile = malloc((size_t)C * W * ptile * sizeof(float));
    float* a = malloc((size_t)rows * W * ptile * sizeof(float));
    float* b = malloc((size_t)rows * W * ptile * sizeof(float));
    int* offs = malloc(sizeof(int) * C);
    int* nds = malloc(sizeof(int) * C);
#pragma omp for schedule(dynamic)
    for (int it = 0; it < ntile; ++it) {
      int p0 = it * ptile;
      int pt = P - p0 < ptile ? P - p0 : ptile;
      load_tile(data, C, W, P, p0, pt, tile);
      float* r = fdmt_tile(tile, C, W, pt, plan, a, b, offs, nds);
      for (int d = 0; d < final_rows; ++d)
        for (int t = 0; t < W; ++t)
          memcpy(out + ((size_t)d * W + t) * P + p0, r + ((size_t)d * W + t) * pt,
                 (size_t)pt * sizeof(float));
    }
    free(tile); free(a); free(b); free(offs); free(nds);
  }
  return final_rows;
}

/*
 * FDMT + boxcar search of the last nb samples of a window [C][W][P].
 *
 * For every pixel and every segment of seg new samples, returns the best S/N
 * over delay rows row0..D, boxcar widths and times in the segment
 * (best_snr [P][nseg], best_info [P][nseg][3] = row, t (window index of the
 * boxcar's last sample), width index). A boxcar value is divided by
 * sigma[row][w]. Values with |box| < clip * sigma also accumulate into
 * sumsq/cnt [D + 1][nw], from which the caller updates sigma for the next block.
 * With detrend set, a linear fit over the window is first removed from each
 * channel of each pixel (detrend_tile).
 *
 * Memory traffic sets the speed, so the transform of a tile runs depth first
 * (fdmt_sub): a subband is merged as soon as both its halves exist, and the
 * small subbands of the lower levels stay in the core's cache instead of
 * making a pass over the whole tile per level. The last merge is not stored:
 * each delay row is formed in a buffer of one row and searched at once, and
 * rows below row0 are not formed at all. The sums are the same as those of
 * fdmt_tile, term by term. The work buffers of each thread are kept between
 * calls (thread_buffer).
 */

/* the plan's merge entries by level (1..nlev) and pair, for fdmt_sub */
typedef struct {
  int C, nlev;
  const int* init_nd; /* [C] */
  const int** pair;   /* pair[first[L] + k]: nd, d_lo[nd + 1], d_up[nd + 1] */
  int first[32];
  int maxrows[32];    /* most rows of a subband at each level */
} fplan_t;

static void fplan_init(fplan_t* fp, const int* plan, int C, const int** pairbuf) {
  fp->C = C;
  fp->init_nd = plan;
  fp->pair = pairbuf;
  int nlev = 0;
  while ((1 << nlev) < C) ++nlev;
  fp->nlev = nlev;
  int m = 0;
  for (int c = 0; c < C; ++c) if (plan[c] + 1 > m) m = plan[c] + 1;
  fp->maxrows[0] = m;
  const int* q = plan + C;
  int idx = 0;
  for (int L = 1; L <= nlev; ++L) {
    fp->first[L] = idx;
    m = 0;
    for (int k = 0; k < (C >> L); ++k) {
      pairbuf[idx++] = q;
      if (q[0] + 1 > m) m = q[0] + 1;
      q += 1 + 2 * (q[0] + 1);
    }
    fp->maxrows[L] = m;
  }
}

/* inside channel c the track spans d + 1 samples (as in fdmt_tile) */
static int fdmt_init_chan(const float* x, int nd, int W, int Pt, float* out) {
  const size_t slab = (size_t)W * Pt;
  memcpy(out, x, slab * sizeof(float));
  for (int d = 1; d <= nd; ++d) {
    float* r = out + (size_t)d * slab;
    const float* rp = r - slab;
    memcpy(r, rp, (size_t)d * Pt * sizeof(float));
    const size_t n = (size_t)(W - d) * Pt;
    float* rd = r + (size_t)d * Pt;
    const float* rpd = rp + (size_t)d * Pt;
    for (size_t i = 0; i < n; ++i) rd[i] = rpd[i] + x[i];
  }
  return nd;
}

/* one merged row: the lower half is reached last (as in fdmt_tile) */
static inline void fdmt_merge_row(const float* l, const float* u, int lo, int W, int Pt,
                                  float* o) {
  const size_t slab = (size_t)W * Pt;
  const size_t head = (size_t)lo * Pt;
  memcpy(o, l, head * sizeof(float));
  const size_t n = slab - head;
  const float* lh = l + head;
  float* oh = o + head;
  for (size_t i = 0; i < n; ++i) oh[i] = lh[i] + u[i];
}

/* the rows of subband k of level L into out [nd + 1][W][Pt]; returns nd.
   slots[2 L'], slots[2 L' + 1]: the halves at level L' < L */
static int fdmt_sub(const fplan_t* fp, int L, int k, const float* tile, int W, int Pt,
                    float* const* slots, float* out) {
  const size_t slab = (size_t)W * Pt;
  if (L == 0) return fdmt_init_chan(tile + (size_t)k * slab, fp->init_nd[k], W, Pt, out);
  float* a = slots[2 * (L - 1)];
  float* b = slots[2 * (L - 1) + 1];
  fdmt_sub(fp, L - 1, 2 * k, tile, W, Pt, slots, a);
  fdmt_sub(fp, L - 1, 2 * k + 1, tile, W, Pt, slots, b);
  const int* q = fp->pair[fp->first[L] + k];
  const int nd = q[0];
  const int* d_lo = q + 1;
  const int* d_up = q + 1 + (nd + 1);
  for (int d = 0; d <= nd; ++d)
    fdmt_merge_row(a + (size_t)d_lo[d] * slab, b + (size_t)d_up[d] * slab, d_lo[d], W, Pt,
                   out + (size_t)d * slab);
  return nd;
}

/* a buffer of at least n floats per thread, kept for the next call */
static float* thread_buffer(size_t n) {
  static __thread float* buf = NULL;
  static __thread size_t size = 0;
  if (n > size) {
    free(buf);
    buf = malloc(n * sizeof(float));
    size = buf ? n : 0;
  }
  return buf;
}

/*
 * The search of one window [C][Wh][P] (a tier's history). The trend
 * (detrend: 1 a straight line, 2 the mean, 0 none) is fitted over all Wh
 * samples; the FDMT and the zero-DM filter then run over the last
 * W = Wh - f0 samples only, so a long detrend span costs memory but almost no
 * transform time. Boxcars end in the last nb samples; every width must be at
 * most W - nb + 1. strides (NULL: all 1): boxcar wi is evaluated at every
 * strides[wi]-th end sample only, which must divide nb. noise 0: the noise
 * sums (sumsq, cnt) are not accumulated and stay zero. Returns D, or -1 for
 * an invalid width, stride or f0.
 */
int skymon_search(const float* data, int C, int Wh, int P, const int* plan, int row0, int nb,
                  const int* widths, int nw, const float* sigma, int seg, float clip,
                  float* best_snr, int* best_info, double* sumsq, int64_t* cnt,
                  int nthreads, int ptile, int detrend, int zerodm, int f0,
                  const int* strides, int noise) {
  int final_rows = 0;
  max_rows(plan, C, &final_rows);
  const int D = final_rows - 1;
  const int nseg = (nb + seg - 1) / seg;
  if (f0 < 0 || f0 >= Wh) return -1;
  const int W = Wh - f0;
  const int t_first = W - nb;
  for (int wi = 0; wi < nw; ++wi) {
    if (widths[wi] < 1 || widths[wi] > t_first + 1) return -1;
    if (strides && (strides[wi] < 1 || nb % strides[wi])) return -1;
  }
  if (ptile <= 0) ptile = 16;
  const int ntile = (P + ptile - 1) / ptile;
  const int** pairbuf = malloc(sizeof(int*) * (C > 1 ? C : 1));
  fplan_t fp;
  fplan_init(&fp, plan, C, pairbuf);
  const size_t slab = (size_t)W * ptile;
  /* per thread: tile, the two halves of every level below the last, one row */
  size_t need = (size_t)C * Wh * ptile + slab + (size_t)(W + 1) * ptile;
  for (int L = 0; L < fp.nlev; ++L) need += 2 * (size_t)fp.maxrows[L] * slab;
  if (fp.nlev == 0) need += (size_t)fp.maxrows[0] * slab;
  for (int i = 0; i < P * nseg; ++i) {
    best_snr[i] = -INFINITY;
    best_info[3 * i] = best_info[3 * i + 1] = best_info[3 * i + 2] = -1;
  }
  memset(sumsq, 0, sizeof(double) * (size_t)final_rows * nw);
  memset(cnt, 0, sizeof(int64_t) * (size_t)final_rows * nw);
#pragma omp parallel num_threads(nthreads)
  {
    float* wsp = thread_buffer(need);
    float* tile = wsp;
    float* row = tile + (size_t)C * Wh * ptile;
    float* cs = row + slab;
    float* slots[64];
    float* s = cs + (size_t)(W + 1) * ptile;
    for (int L = 0; L < fp.nlev; ++L) {
      slots[2 * L] = s;
      s += (size_t)fp.maxrows[L] * slab;
      slots[2 * L + 1] = s;
      s += (size_t)fp.maxrows[L] * slab;
    }
    float* single = s; /* C == 1: the rows of the only channel */
    double* ssq = calloc((size_t)final_rows * nw, sizeof(double));
    int64_t* nn = calloc((size_t)final_rows * nw, sizeof(int64_t));
    /* best per segment and pixel, [nseg][ptile] so the pixel loop is contiguous */
    float* bb = malloc((size_t)ptile * nseg * sizeof(float));
    int* bd = malloc((size_t)ptile * nseg * sizeof(int));
    int* bt = malloc((size_t)ptile * nseg * sizeof(int));
    int* bw = malloc((size_t)ptile * nseg * sizeof(int));
    /* per-pixel noise sums over one row and width (at most nb terms, each
       below clip^2 sigma^2): float is ample and vectorises twice as wide */
    float* accp = malloc((size_t)ptile * sizeof(float));
    int32_t* nap = malloc((size_t)ptile * sizeof(int32_t));
    double* dsx = malloc((size_t)ptile * sizeof(double));
    double* dstx = malloc((size_t)ptile * sizeof(double));
#pragma omp for schedule(dynamic)
    for (int it = 0; it < ntile; ++it) {
      int p0 = it * ptile;
      int pt = P - p0 < ptile ? P - p0 : ptile;
      const size_t sl = (size_t)W * pt;
      load_tile(data, C, Wh, P, p0, pt, tile);
      if (detrend == 1) detrend_tile(tile, C, Wh, pt, dsx, dstx);
      else if (detrend == 2) demean_tile(tile, C, Wh, pt, dsx);
      if (f0 > 0) /* keep the last W samples of every channel, packed [C][W][pt] */
        for (int c = 0; c < C; ++c)
          memmove(tile + (size_t)c * W * pt, tile + ((size_t)c * Wh + f0) * pt,
                  (size_t)W * pt * sizeof(float));
      if (zerodm) zerodm_tile(tile, C, W, pt);
      for (int i = 0; i < pt * nseg; ++i) {
        bb[i] = -INFINITY;
        bd[i] = bt[i] = bw[i] = -1;
      }
      const float* ra = NULL;
      const float* rb = NULL;
      const int* q = NULL;
      if (fp.nlev == 0) {
        fdmt_init_chan(tile, fp.init_nd[0], W, pt, single);
      } else {
        float* a = slots[2 * (fp.nlev - 1)];
        float* b = slots[2 * (fp.nlev - 1) + 1];
        fdmt_sub(&fp, fp.nlev - 1, 0, tile, W, pt, slots, a);
        fdmt_sub(&fp, fp.nlev - 1, 1, tile, W, pt, slots, b);
        ra = a;
        rb = b;
        q = fp.pair[fp.first[fp.nlev]];
      }
      for (int d = row0; d <= D; ++d) {
        const float* r;
        if (fp.nlev == 0) {
          r = single + (size_t)d * sl;
        } else {
          const int lo = q[1 + d], up = q[1 + (D + 1) + d];
          fdmt_merge_row(ra + (size_t)lo * sl, rb + (size_t)up * sl, lo, W, pt, row);
          r = row;
        }
        /* prefix sums over time, per pixel: cs[t + 1] = sum r[0..t] */
        for (int p = 0; p < pt; ++p) cs[p] = 0.f;
        for (int t = 0; t < W; ++t)
          for (int p = 0; p < pt; ++p)
            cs[(size_t)(t + 1) * pt + p] = cs[(size_t)t * pt + p] + r[(size_t)t * pt + p];
        for (int wi = 0; wi < nw; ++wi) {
          const int w = widths[wi];
          const float sg_ = sigma[d * nw + wi];
          const float inv = sg_ > 0 ? 1.f / sg_ : 0.f;
          const float lim = clip * sg_;
          /* a wide boxcar may be evaluated every stride samples, ending on
             the last sample of the window (strides divide nb) */
          const int st = strides ? strides[wi] : 1;
          for (int p = 0; p < pt; ++p) { accp[p] = 0; nap[p] = 0; }
          for (int t = t_first + st - 1; t < W; t += st) {
            const int sg = (t - t_first) / seg;
            const float* restrict hi = cs + (size_t)(t + 1) * pt;
            const float* restrict lo = cs + (size_t)(t + 1 - w) * pt;
            float* restrict b1 = bb + (size_t)sg * pt;
            int* restrict b2 = bd + (size_t)sg * pt;
            int* restrict b3 = bt + (size_t)sg * pt;
            int* restrict b4 = bw + (size_t)sg * pt;
            if (noise) {
              for (int p = 0; p < pt; ++p) {
                const float v = hi[p] - lo[p];
                const int in = fabsf(v) < lim;
                accp[p] += in ? v * v : 0.f;
                nap[p] += in;
                const float snr = v * inv;
                const int better = snr > b1[p];
                b1[p] = better ? snr : b1[p];
                b2[p] = better ? d : b2[p];
                b3[p] = better ? t : b3[p];
                b4[p] = better ? wi : b4[p];
              }
            } else {
              for (int p = 0; p < pt; ++p) {
                const float snr = (hi[p] - lo[p]) * inv;
                const int better = snr > b1[p];
                b1[p] = better ? snr : b1[p];
                b2[p] = better ? d : b2[p];
                b3[p] = better ? t : b3[p];
                b4[p] = better ? wi : b4[p];
              }
            }
          }
          double acc = 0;
          int64_t na = 0;
          for (int p = 0; p < pt; ++p) { acc += (double)accp[p]; na += nap[p]; }
          ssq[d * nw + wi] += acc;
          nn[d * nw + wi] += na;
        }
      }
      for (int p = 0; p < pt; ++p)
        for (int sgi = 0; sgi < nseg; ++sgi) {
          int k = sgi * pt + p, g = (p0 + p) * nseg + sgi;
          best_snr[g] = bb[k];
          best_info[3 * g] = bd[k];
          best_info[3 * g + 1] = bt[k];
          best_info[3 * g + 2] = bw[k];
        }
    }
#pragma omp critical
    {
      for (int i = 0; i < final_rows * nw; ++i) { sumsq[i] += ssq[i]; cnt[i] += nn[i]; }
    }
    free(ssq); free(nn); free(bb); free(bd); free(bt); free(bw); free(accp); free(nap);
    free(dsx); free(dstx);
  }
  free(pairbuf);
  return D;
}

/*
 * Copy rows src..src + n - 1 of a window [C][W][P] to rows dst..dst + n - 1,
 * in parallel over channels: the shift of a searched window (Searcher.finish_block).
 */
void skymon_shift(float* hist, int C, int W, int P, int dst, int src, int n, int nthreads) {
#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int c = 0; c < C; ++c) {
    float* h = hist + (size_t)c * W * P;
    memmove(h + (size_t)dst * P, h + (size_t)src * P, (size_t)n * P * sizeof(float));
  }
}

/*
 * Regridding of one image onto the tracked sky pixels by a weighted sum of
 * ntaps image pixels (16 for bicubic interpolation).
 * img [C][NN] (FITS order, y major), idx/w [P][ntaps] -> out [C][P].
 * Pixels with all weights zero come out as zero.
 */
void skymon_regrid(const float* img, int C, int NN, const int* idx, const float* w, int P,
                   int ntaps, float* out, int nthreads) {
#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int c = 0; c < C; ++c) {
    const float* im = img + (size_t)c * NN;
    float* o = out + (size_t)c * P;
    for (int p = 0; p < P; ++p) {
      const int* ix = idx + (size_t)ntaps * p;
      const float* wx = w + (size_t)ntaps * p;
      float v = 0.f;
      for (int k = 0; k < ntaps; ++k) v += wx[k] * im[ix[k]];
      o[p] = v;
    }
  }
}

/*
 * Bicubic taps of every slot for one image (geometry.SkyGrid.taps): the
 * position xy = xy_a + f (xy_b - xy_a) (1-based FITS pixels, [P][2], NaN for
 * a free slot), the flat indices idx [P][16] into a [grid][grid] image and the
 * weights w [P][16] of Keys' cubic convolution (a = -0.75); a point within two
 * pixels of the edge, or not finite, gets zero weights. The arithmetic is that
 * of geometry.bicubic and geometry._keys in double precision, operation by
 * operation, so this function is compiled without the fast-math options.
 */
#pragma GCC push_options
#pragma GCC optimize("-fno-fast-math", "-ffp-contract=off", "-fno-unsafe-math-optimizations")
static void keys4(double f, double* wt) {
  const double a = -0.75;
  const double t[4] = {1 + f, f, 1 - f, 2 - f};
  for (int k = 0; k < 4; ++k) {
    const double t2 = t[k] * t[k], t3 = pow(t[k], 3.0);
    const double near = (a + 2) * t3 - (a + 3) * t2 + 1;
    const double far = a * t3 - (5 * a) * t2 + (8 * a) * t[k] - 4 * a;
    wt[k] = t[k] <= 1 ? near : far;
  }
}

void skymon_taps(const double* xy_a, const double* xy_b, double f, int P, int grid,
                 double* xy, int32_t* idx, float* w, int nthreads) {
#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int p = 0; p < P; ++p) {
    const double xx = xy_a[2 * p] + f * (xy_b[2 * p] - xy_a[2 * p]);
    const double yy = xy_a[2 * p + 1] + f * (xy_b[2 * p + 1] - xy_a[2 * p + 1]);
    xy[2 * p] = xx;
    xy[2 * p + 1] = yy;
    const double x = xx - 1.0, y = yy - 1.0;
    const double fx0 = floor(x), fy0 = floor(y);
    const int ok = isfinite(fx0) && isfinite(fy0) && fx0 >= 1 && fy0 >= 1 &&
                   fx0 < grid - 2 && fy0 < grid - 2;
    const int64_t x0 = ok ? (int64_t)fx0 : 1, y0 = ok ? (int64_t)fy0 : 1;
    double wx[4], wy[4];
    keys4(ok ? x - (double)x0 : 0.0, wx);
    keys4(ok ? y - (double)y0 : 0.0, wy);
    int32_t* ix = idx + (size_t)16 * p;
    float* wp = w + (size_t)16 * p;
    for (int j = 0; j < 4; ++j)
      for (int i = 0; i < 4; ++i) {
        ix[4 * j + i] = (int32_t)((y0 + j - 1) * grid + (x0 + i - 1));
        wp[4 * j + i] = ok ? (float)(wy[j] * wx[i]) : 0.f;
      }
  }
}
#pragma GCC pop_options

/* ------------------------------------------------------------ ingest --- */

static float select_kth(float* a, int n, int k) {
  int lo = 0, hi = n - 1;
  while (lo < hi) {
    float pivot = a[(lo + hi) / 2];
    int i = lo, j = hi;
    while (i <= j) {
      while (a[i] < pivot) ++i;
      while (a[j] > pivot) --j;
      if (i <= j) { float t = a[i]; a[i] = a[j]; a[j] = t; ++i; --j; }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else return a[k];
  }
  return a[k];
}

/*
 * Normalise one regridded image in place, flag interference, and accumulate
 * the raw values.
 *
 * raw [C][P]: sky values (regrid output); on return x = normalised.
 * mu, var [C][P]: running mean and variance (exponential, per pixel/channel).
 *   The variance is half the mean square difference between consecutive
 *   images, which a slow trend (a source rising through the primary beam)
 *   does not inflate. prev [C][P] holds the previous raw value.
 * age [P]: images since the pixel started being tracked (0 = not tracked);
 *   updated here for live pixels. While age <= warmup the statistics use a
 *   running average and the output is zero.
 * acc [C][P] += raw (for the spectrogram accumulation) when acc != NULL.
 * sub [nsub]: slots used for the common-mode median of each channel.
 * a_mu, a_var: exponential weights per image after warm-up.
 * clip: a cell (pixel and channel) with |x| above this is set to zero, unless
 *   it is sky-shaped (nbr below).
 * nbr [P][nnbr] (may be NULL): the slots of each pixel's sky neighbours (-1:
 *   none). A cell beyond the clip is kept when at least nbr_min of its
 *   neighbours, in the same channel and image, deviate with the same sign by
 *   at least nbr_frac of its own deviation. Anything from the sky reaches the
 *   image through the synthesised beam, which is wider than a sky pixel
 *   (pixels of at most half the beam: a neighbour sees 0.5-0.8 of a point
 *   source's peak), so a bright burst lights its neighbours, while a glitch of
 *   a single cell does not. A kept cell is still withheld from the
 *   statistics, like a clipped one.
 * med_out [C]: the common-mode median of each channel (may be NULL).
 *
 * Interference flagging per channel. After the common mode is removed, the
 * spread of x over the subset of warm pixels (1.4826 times the median
 * absolute deviation) is close to 1 for noise, whatever the sky. A carrier
 * that reaches much of the sky in one channel (aircraft or meteor scatter of
 * a distant transmitter, seen directly and through the sidelobes) widens it,
 * while a burst, confined to a beam, leaves it unchanged. The spread is
 * compared with that of the band: the median spread over the channels, or 1
 * if that is smaller. A rise of noise over the whole band (a thunderstorm, a
 * solar burst, a change of gain) widens every channel alike, and is left to
 * the search and to the conditions of the band instead of blanking the band.
 * A channel whose spread exceeds chan_flag_sigma times that of the band is
 * set to zero for every pixel in this image and reported in flags_out [C]
 * (1 = flagged). The spread needs at least 64 warm pixels in the subset;
 * with fewer no channel is flagged.
 * chan_mask [C] (may be NULL): channels set to zero regardless (the
 * persistent mask kept by the caller). They are still measured and flagged,
 * so the caller can tell when the interference has stopped.
 * spread_out [C] (may be NULL): the measured spread of each channel.
 *
 * Statistics. The raw value of a cell is compared with the statistics of the
 * previous images first; the statistics are updated only afterwards, and not
 * at all for a cell whose output is set to zero by a channel flag or by the
 * clip: interference must not enter the mean and variance against which the
 * next images are measured. held [C][P] counts the consecutive images for
 * which a cell's update was withheld; after max_hold images the cell is
 * updated again (with deviations limited to 5 sigma), so that a lasting
 * change of level, not interference, is followed in the end instead of
 * blanking the cell for good. A cell whose update is withheld keeps its
 * previous raw value, so the next successive difference spans the gap.
 * Without held (NULL) every cell is updated, as before flagging existed.
 * A channel set to zero only by chan_mask updates its statistics normally.
 * Returns the number of live pixels.
 */
int skymon_normalise(float* raw, int C, int P, float* mu, float* var, float* prev, int32_t* age,
                     const uint8_t* live, float* acc, const int32_t* sub, int nsub,
                     int warmup, float a_mu, float a_var, float clip, float* med_out,
                     float chan_flag_sigma, const uint8_t* chan_mask, uint8_t* flags_out,
                     float* spread_out, uint16_t* held, int max_hold,
                     const int32_t* nbr, int nnbr, int nbr_min, float nbr_frac, int nthreads) {
  int nlive = 0;
  for (int p = 0; p < P; ++p) {
    if (live[p]) { age[p] += 1; ++nlive; } else age[p] = 0;
  }
  if (max_hold > 65535) max_hold = 65535;
  float* meds = malloc(sizeof(float) * (C > 0 ? C : 1));
  float* spreads = malloc(sizeof(float) * (C > 0 ? C : 1));
  int* nsubs = malloc(sizeof(int) * (C > 0 ? C : 1));
  /* the raw values of every cell, kept from the first pass to the second, and
     the clip state; kept for the next call (thread_buffer), since a fresh
     buffer of this size costs page faults on every image */
  const size_t ncell = (size_t)C * P > 0 ? (size_t)C * P : 1;
  float* rvall = thread_buffer(ncell + (ncell + sizeof(float) - 1) / sizeof(float));
  uint8_t* ovall = (uint8_t*)(rvall + ncell);
  float band_spread = 1.f;
#pragma omp parallel num_threads(nthreads)
  {
    float* tmp = malloc(sizeof(float) * (nsub > 0 ? nsub : 1));
#pragma omp for schedule(static)
    for (int c = 0; c < C; ++c) {
      float* x = raw + (size_t)c * P;
      float* m = mu + (size_t)c * P;
      float* v = var + (size_t)c * P;
      float* pv = prev + (size_t)c * P;
      uint16_t* hd = held ? held + (size_t)c * P : NULL;
      float* ac = acc ? acc + (size_t)c * P : NULL;
      float* rv = rvall + (size_t)c * P;
      /* pass 1: deviations from the statistics of the previous images; the
         statistics of pixels still warming up are updated here */
      for (int p = 0; p < P; ++p) {
        if (!live[p]) { x[p] = 0.f; continue; }
        float r = x[p];
        rv[p] = r;
        if (ac) ac[p] += r;
        int a = age[p];
        if (a == 1) { m[p] = r; v[p] = 0.f; pv[p] = r; x[p] = 0.f; if (hd) hd[p] = 0; continue; }
        if (a <= warmup) {
          float wm = 1.f / a, wv = 1.f / (a - 1);
          float d = r - m[p], dd = r - pv[p];
          pv[p] = r;
          v[p] += wv * (0.5f * dd * dd - v[p]);
          m[p] += wm * d;
          x[p] = 0.f;
          continue;
        }
        x[p] = v[p] > 0.f ? (r - m[p]) / sqrtf(v[p]) : 0.f;
      }
      /* common mode: median over the subset of warm pixels, then the spread
         about it (the median absolute deviation, on the same copy) */
      int n = 0;
      for (int i = 0; i < nsub; ++i) {
        int s = sub[i];
        if (live[s] && age[s] > warmup) tmp[n++] = x[s];
      }
      float med = n > 8 ? select_kth(tmp, n, n / 2) : 0.f;
      float spread = 0.f;
      if (n >= 64) {
        for (int i = 0; i < n; ++i) tmp[i] = fabsf(tmp[i] - med);
        spread = 1.4826f * select_kth(tmp, n, n / 2);
      }
      meds[c] = med;
      spreads[c] = spread;
      nsubs[c] = n;
    }
    /* the spread of the band: the median over the channels measured */
#pragma omp single
    {
      int k = 0;
      float* t = malloc(sizeof(float) * (C > 0 ? C : 1));
      for (int c = 0; c < C; ++c)
        if (nsubs[c] >= 64) t[k++] = spreads[c];
      if (k > 0) {
        float b = select_kth(t, k, k / 2);
        if (b > band_spread) band_spread = b;
      }
      free(t);
    }
#pragma omp for schedule(static)
    for (int c = 0; c < C; ++c) {
      float* x = raw + (size_t)c * P;
      float* m = mu + (size_t)c * P;
      float* v = var + (size_t)c * P;
      float* pv = prev + (size_t)c * P;
      uint16_t* hd = held ? held + (size_t)c * P : NULL;
      const float* rv = rvall + (size_t)c * P;
      uint8_t* ov = ovall + (size_t)c * P;
      const float med = meds[c];
      const int flag = nsubs[c] >= 64 && spreads[c] > chan_flag_sigma * band_spread;
      const int zero = flag || (chan_mask && chan_mask[c]);
      if (med_out) med_out[c] = med;
      if (spread_out) spread_out[c] = spreads[c];
      if (flags_out) flags_out[c] = (uint8_t)flag;
      /* the clip, decided on the values before any is set to zero: 1 = clipped,
         2 = beyond the clip but sky-shaped (kept, withheld from the statistics) */
      for (int p = 0; p < P; ++p) {
        ov[p] = 0;
        if (!live[p] || age[p] <= warmup) continue;
        const float y = x[p] - med;
        if (!(fabsf(y) > clip)) continue;
        ov[p] = 1;
        if (!nbr) continue;
        const float lim = nbr_frac * fabsf(y);
        int k = 0;
        for (int j = 0; j < nnbr; ++j) {
          const int q = nbr[(size_t)p * nnbr + j];
          if (q < 0 || q >= P || !live[q] || age[q] <= warmup) continue;
          const float yq = x[q] - med;
          if ((y > 0 ? yq : -yq) >= lim) ++k;
        }
        if (k >= nbr_min) ov[p] = 2;
      }
      /* pass 2: output, then the statistics of the cells that are kept */
      for (int p = 0; p < P; ++p) {
        if (!live[p] || age[p] <= warmup) continue;
        float y = x[p] - med;
        const int over = ov[p] != 0;
        x[p] = (zero || ov[p] == 1) ? 0.f : y;
        if (hd) {
          if (!(flag || over)) hd[p] = 0;
          else if (hd[p] < max_hold) { hd[p] += 1; continue; }
        }
        /* update the statistics, with deviations limited to 5 sigma so that
           what the flags miss (weaker interference, bursts) does not
           inflate them */
        float r = rv[p];
        float d = r - m[p];
        float du = d, dd = r - pv[p];
        pv[p] = r;
        if (v[p] > 0.f) {
          float lim = 5.f * sqrtf(v[p]);
          du = d > lim ? lim : (d < -lim ? -lim : d);
          float lim2 = 1.41421356f * lim;
          dd = dd > lim2 ? lim2 : (dd < -lim2 ? -lim2 : dd);
        }
        v[p] += a_var * (0.5f * dd * dd - v[p]);
        m[p] += a_mu * du;
      }
    }
    free(tmp);
  }
  free(meds);
  free(spreads);
  free(nsubs);
  return nlive;
}

/*
 * Append one sample to a tier: y = x - base (high-pass), base += a_hp (x - base),
 * hist[c][pos][p] = y; dec[c][p] += x (input of the next tier).
 * hist is [C][W][P]; dec may be NULL.
 */
void skymon_tier_push(const float* x, int C, int P, float* base, float* hist, int W, int pos,
                      float a_hp, float* dec, int nthreads) {
#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int c = 0; c < C; ++c) {
    const float* xc = x + (size_t)c * P;
    float* b = base + (size_t)c * P;
    float* h = hist + ((size_t)c * W + pos) * P;
    float* dc = dec ? dec + (size_t)c * P : NULL;
    for (int p = 0; p < P; ++p) {
      float v = xc[p];
      float y = v - b[p];
      b[p] += a_hp * y;
      h[p] = y;
      if (dc) dc[p] += v;
    }
  }
}

/*
 * One image into the tiers it reaches, in one call: x into tier 0 as
 * skymon_tier_push does; for j < ntier - 1, tier j's decimation sum is then
 * complete, and dec_j * scale (scale = 1/sqrt(2)) is pushed into tier j + 1
 * while dec_j is set to zero. The last tier reached adds its input to its own
 * sum when dec[ntier - 1] is not NULL. The same operations as successive
 * calls of skymon_tier_push (Searcher._push_tier), one channel at a time.
 * base, hist, dec: arrays of ntier pointers; W, pos, a_hp: [ntier] (each tier's
 * high-pass weight, 1 / hp_tau).
 */
void skymon_tier_cascade(const float* x, int C, int P, int ntier, float** base, float** hist,
                         const int* W, const int* pos, float** dec, const float* a_hp,
                         float scale, int nthreads) {
#pragma omp parallel num_threads(nthreads)
  {
    float* tmp = thread_buffer((size_t)P > 0 ? (size_t)P : 1);
#pragma omp for schedule(static)
    for (int c = 0; c < C; ++c) {
      const float* xc = x + (size_t)c * P;
      for (int j = 0; j < ntier; ++j) {
        float* b = base[j] + (size_t)c * P;
        float* h = hist[j] + ((size_t)c * W[j] + pos[j]) * P;
        float* dc = dec[j] ? dec[j] + (size_t)c * P : NULL;
        const float a = a_hp[j];
        for (int p = 0; p < P; ++p) {
          float v = xc[p];
          float y = v - b[p];
          b[p] += a * y;
          h[p] = y;
          if (dc) dc[p] += v;
        }
        if (j + 1 < ntier) {
          for (int p = 0; p < P; ++p) {
            tmp[p] = dc[p] * scale;
            dc[p] = 0.f;
          }
          xc = tmp;
        }
      }
    }
  }
}
