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
 */
int skymon_search(const float* data, int C, int W, int P, const int* plan, int row0, int nb,
                  const int* widths, int nw, const float* sigma, int seg, float clip,
                  float* best_snr, int* best_info, double* sumsq, int64_t* cnt,
                  int nthreads, int ptile, int detrend, int zerodm) {
  int final_rows = 0;
  const int rows = max_rows(plan, C, &final_rows);
  const int D = final_rows - 1;
  const int nseg = (nb + seg - 1) / seg;
  const int t_first = W - nb;
  if (ptile <= 0) ptile = 16;
  const int ntile = (P + ptile - 1) / ptile;
  for (int i = 0; i < P * nseg; ++i) {
    best_snr[i] = -INFINITY;
    best_info[3 * i] = best_info[3 * i + 1] = best_info[3 * i + 2] = -1;
  }
  memset(sumsq, 0, sizeof(double) * (size_t)final_rows * nw);
  memset(cnt, 0, sizeof(int64_t) * (size_t)final_rows * nw);
#pragma omp parallel num_threads(nthreads)
  {
    float* tile = malloc((size_t)C * W * ptile * sizeof(float));
    float* a = malloc((size_t)rows * W * ptile * sizeof(float));
    float* b = malloc((size_t)rows * W * ptile * sizeof(float));
    int* offs = malloc(sizeof(int) * C);
    int* nds = malloc(sizeof(int) * C);
    float* cs = malloc((size_t)(W + 1) * ptile * sizeof(float));
    double* ssq = calloc((size_t)final_rows * nw, sizeof(double));
    int64_t* nn = calloc((size_t)final_rows * nw, sizeof(int64_t));
    float* lb = malloc((size_t)ptile * nseg * sizeof(float));
    int* li = malloc((size_t)ptile * nseg * 3 * sizeof(int));
    double* dsx = malloc((size_t)ptile * sizeof(double));
    double* dstx = malloc((size_t)ptile * sizeof(double));
#pragma omp for schedule(dynamic)
    for (int it = 0; it < ntile; ++it) {
      int p0 = it * ptile;
      int pt = P - p0 < ptile ? P - p0 : ptile;
      load_tile(data, C, W, P, p0, pt, tile);
      if (detrend) detrend_tile(tile, C, W, pt, dsx, dstx);
      if (zerodm) zerodm_tile(tile, C, W, pt);
      float* r = fdmt_tile(tile, C, W, pt, plan, a, b, offs, nds);
      for (int i = 0; i < pt * nseg; ++i) {
        lb[i] = -INFINITY;
        li[3 * i] = li[3 * i + 1] = li[3 * i + 2] = -1;
      }
      for (int d = row0; d <= D; ++d) {
        const float* row = r + (size_t)d * W * pt;
        /* prefix sums over time, per pixel: cs[t + 1] = sum row[0..t] */
        for (int p = 0; p < pt; ++p) cs[p] = 0.f;
        for (int t = 0; t < W; ++t)
          for (int p = 0; p < pt; ++p)
            cs[(size_t)(t + 1) * pt + p] = cs[(size_t)t * pt + p] + row[(size_t)t * pt + p];
        for (int wi = 0; wi < nw; ++wi) {
          const int w = widths[wi];
          const float s = sigma[d * nw + wi];
          const float inv = s > 0 ? 1.f / s : 0.f;
          const float lim = clip * s;
          double acc = 0;
          int64_t na = 0;
          for (int t = t_first; t < W; ++t) {
            const int sg = (t - t_first) / seg;
            const float* hi = cs + (size_t)(t + 1) * pt;
            const float* lo = cs + (size_t)(t + 1 - w) * pt;
            for (int p = 0; p < pt; ++p) {
              float v = hi[p] - lo[p];
              if (fabsf(v) < lim) { acc += (double)v * v; ++na; }
              float snr = v * inv;
              int k = p * nseg + sg;
              if (snr > lb[k]) {
                lb[k] = snr;
                li[3 * k] = d; li[3 * k + 1] = t; li[3 * k + 2] = wi;
              }
            }
          }
          ssq[d * nw + wi] += acc;
          nn[d * nw + wi] += na;
        }
      }
      for (int p = 0; p < pt; ++p)
        for (int sgi = 0; sgi < nseg; ++sgi) {
          int k = p * nseg + sgi, g = (p0 + p) * nseg + sgi;
          best_snr[g] = lb[k];
          best_info[3 * g] = li[3 * k];
          best_info[3 * g + 1] = li[3 * k + 1];
          best_info[3 * g + 2] = li[3 * k + 2];
        }
    }
#pragma omp critical
    {
      for (int i = 0; i < final_rows * nw; ++i) { sumsq[i] += ssq[i]; cnt[i] += nn[i]; }
    }
    free(tile); free(a); free(b); free(offs); free(nds); free(cs); free(ssq); free(nn);
    free(lb); free(li); free(dsx); free(dstx);
  }
  return D;
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
 * Normalise one regridded image in place and accumulate the raw values.
 *
 * raw [C][P]: sky values (bilinear regrid output); on return x = normalised.
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
 * clip: |x| above this is set to zero.
 * Returns the number of live pixels.
 */
int skymon_normalise(float* raw, int C, int P, float* mu, float* var, float* prev, int32_t* age,
                     const uint8_t* live, float* acc, const int32_t* sub, int nsub,
                     int warmup, float a_mu, float a_var, float clip, float* med_out,
                     int nthreads) {
  int nlive = 0;
  for (int p = 0; p < P; ++p) {
    if (live[p]) { age[p] += 1; ++nlive; } else age[p] = 0;
  }
#pragma omp parallel num_threads(nthreads)
  {
    float* tmp = malloc(sizeof(float) * (nsub > 0 ? nsub : 1));
#pragma omp for schedule(static)
    for (int c = 0; c < C; ++c) {
      float* x = raw + (size_t)c * P;
      float* m = mu + (size_t)c * P;
      float* v = var + (size_t)c * P;
      float* pv = prev + (size_t)c * P;
      float* ac = acc ? acc + (size_t)c * P : NULL;
      for (int p = 0; p < P; ++p) {
        if (!live[p]) { x[p] = 0.f; continue; }
        float r = x[p];
        if (ac) ac[p] += r;
        int a = age[p];
        float wm, wv;
        if (a <= warmup) { wm = 1.f / a; wv = a > 1 ? 1.f / (a - 1) : 0.f; }
        else { wm = a_mu; wv = a_var; }
        float d = r - m[p];
        if (a == 1) { m[p] = r; v[p] = 0.f; pv[p] = r; x[p] = 0.f; continue; }
        /* update the statistics, with deviations limited to 5 sigma after
           warm-up so that RFI and bursts do not inflate them */
        float du = d, dd = r - pv[p];
        pv[p] = r;
        if (a > warmup && v[p] > 0.f) {
          float lim = 5.f * sqrtf(v[p]);
          du = d > lim ? lim : (d < -lim ? -lim : d);
          float lim2 = 1.41421356f * lim;
          dd = dd > lim2 ? lim2 : (dd < -lim2 ? -lim2 : dd);
        }
        v[p] += wv * (0.5f * dd * dd - v[p]);
        m[p] += wm * du;
        x[p] = (a > warmup && v[p] > 0.f) ? d / sqrtf(v[p]) : 0.f;
      }
      /* common mode: median over the subset of warm pixels */
      int n = 0;
      for (int i = 0; i < nsub; ++i) {
        int s = sub[i];
        if (live[s] && age[s] > warmup) tmp[n++] = x[s];
      }
      float med = n > 8 ? select_kth(tmp, n, n / 2) : 0.f;
      if (med_out) med_out[c] = med;
      for (int p = 0; p < P; ++p) {
        if (!live[p] || age[p] <= warmup) continue;
        float y = x[p] - med;
        x[p] = fabsf(y) > clip ? 0.f : y;
      }
    }
    free(tmp);
  }
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
