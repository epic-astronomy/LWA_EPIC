// Correctness tests for the epic++ imaging kernel (BlockEpicImager).
//
// Each test drives the production kernel through ImagerHarness with simulated
// LWA-SV voltages and checks a property of the output image:
//   gcf_normalised      per-antenna kernel weights sum to 1 (ComputeGcfElements)
//   point_positions     point sources land at their expected pixels
//   run_to_run_noise    two identical gulps agree (half-precision atomics and
//                       16-bit accumulation make the kernel non-deterministic)
//   streams_match_single_launch  the per-stream split of ProcessGulp
//                       (chan_offset > 0) gives the same image as one launch
//   wproj_identity      complex path with real weights == real path, to
//                       half-precision rounding
//   weight_modes_agree  every kernel-weight storage mode (global, global-half2,
//                       shared, registers, registers-half2) gives the same image
//   flux_vs_elevation   equal sources from zenith to 20 deg elevation have equal
//                       peaks after the gridding correction (within 5%)
//   wproj_corrects_w    with the full stand heights in the data (zenith
//                       phasing), w-projected kernels remove the position error
//   gulp_accumulation   a second gulp with is_first_gulp=false adds to the first,
//                       within the run-to-run noise
//   acorr_single_stand  with one stand imaged per channel the image is that
//                       stand's auto-correlation alone; --remove_autocorr must
//                       remove it (XX, YY, U and V; real and w-projected
//                       kernels; complex, unequal pol gains)
//   stokes_u            a source with identical X and Y voltages has
//                       U = Re(XY*) = XX and V = Im(XY*) = 0 at its peak
//   chunked_launch_matches  a gulp imaged in launches of 37 sequences (stand
//                       powers carried between them for the auto-correlation
//                       removal) equals one launch (32-bit accumulation)
//   accum16_chunked     16-bit accumulation over a 1000-sequence gulp in
//                       launches of 100 sequences is within 1% of 32-bit
//                       (one launch: ~25% low)
//   acorr_pedestal      sources plus noise on all stands: removal takes the
//                       smooth background to ~0 and leaves source peaks above
//                       the background unchanged (within 3%)
//
// Exit code is non-zero if any test fails. Run: ./test_imager [--quick]
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "epic_harness.cuh"

using namespace epic_test;

namespace {

bool g_quick = false;

ImagerConfig SmallConfig() {
  ImagerConfig c;
  c.nseq = g_quick ? 100 : 400;
  c.nchan = 8;
  c.nstreams = 2;
  c.support = 5;  // production --support 5
  return c;
}

struct Offsets {
  double mean = 0, max = 0;
  int n = 0;
};

// Peak offsets (pixels) from the expected positions, averaged over channels
Offsets MeasureOffsets(ImagerHarness& h, const std::vector<Source>& srcs) {
  auto out = h.DownloadOut();
  Offsets o;
  for (int c = 0; c < h.config().nchan; ++c) {
    auto img = h.ImageXX(out, c);
    for (const auto& s : srcs) {
      double ex, ey, px, py;
      ImagerHarness::ExpectedPixel(s.l, s.m, &ex, &ey);
      if (!FindPeak(img, ex, ey, 3, &px, &py)) continue;
      double d = std::hypot(px - ex, py - ey);
      o.mean += d;
      o.max = std::max(o.max, d);
      o.n++;
    }
  }
  if (o.n) o.mean /= o.n;
  return o;
}

struct Diff {
  double max_rel, rms_rel;
};

// XX difference between two outputs relative to the peak of a
Diff Compare(const std::vector<float>& a, const std::vector<float>& b, double scale_b = 1.0) {
  double peak = 0, mx = 0, ss = 0;
  size_t n = 0;
  for (size_t i = 0; i < a.size(); i += 4) peak = std::max(peak, double(std::abs(a[i])));
  for (size_t i = 0; i < a.size(); i += 4, ++n) {
    double d = std::abs(double(b[i]) - scale_b * a[i]);
    mx = std::max(mx, d);
    ss += d * d;
  }
  return {mx / peak, std::sqrt(ss / n) / peak};
}

// Two identical launches: the kernel's own run-to-run noise (half-precision
// atomics in the gridder; bfloat16 on-chip accumulation with --accum_16bit,
// see f16_accum_t in MOFF_cu_handler.cu)
Diff RunToRun(ImagerHarness& h) {
  h.LaunchGulp(true, false);
  h.Sync();
  auto a = h.DownloadOut();
  h.LaunchGulp(true, false);
  h.Sync();
  return Compare(a, h.DownloadOut());
}

Source Dir(double az_deg, double el_deg, double amp, bool with_w) {
  double az = az_deg * M_PI / 180, el = el_deg * M_PI / 180;
  return {std::cos(el) * std::sin(az), std::cos(el) * std::cos(az), amp, with_w};
}

bool TestGcfNormalised() {
  ImagerHarness h(SmallConfig());
  auto e = h.GcfElements();
  int nel = h.config().support * h.config().support;
  double worst = 0;
  for (size_t i = 0; i < e.size() / nel; ++i) {
    double s = 0;
    for (int k = 0; k < nel; ++k) s += e[i * nel + k];
    worst = std::max(worst, std::abs(s - 1.0));
  }
  std::printf("    max |sum(weights) - 1| = %.2e\n", worst);
  return worst < 1e-4;
}

bool TestPointPositions() {
  ImagerHarness h(SmallConfig());
  // Grid-frame directions, no w-term in the data: positions must be exact
  std::vector<Source> srcs = {Dir(0, 90, 2.0, false), Dir(30, 60, 2.0, false),
                              Dir(150, 50, 2.0, false), Dir(250, 45, 2.0, false),
                              Dir(330, 35, 2.0, false)};
  h.SimulateSources(srcs, 0.5);
  h.UploadFEng();
  h.LaunchGulp(true, false);
  h.Sync();
  auto o = MeasureOffsets(h, srcs);
  std::printf("    %d peaks, offset mean %.3f px, max %.3f px\n", o.n, o.mean, o.max);
  return o.n == int(srcs.size()) * h.config().nchan && o.mean < 0.15 && o.max < 0.35;
}

bool TestRunToRunNoise() {
  // Informational baseline: how far two identical gulps differ
  bool ok = true;
  for (bool acc16 : {true, false}) {
    auto cfg = SmallConfig();
    cfg.accum16 = acc16;
    ImagerHarness h(cfg);
    h.SimulateSources({Dir(0, 90, 2.0, true), Dir(120, 40, 2.0, true)}, 0.5);
    h.UploadFEng();
    auto d = RunToRun(h);
    std::printf("    accum %s: run-to-run max %.2e, rms %.2e of peak\n", acc16 ? "16-bit" : "32-bit",
                d.max_rel, d.rms_rel);
    ok &= d.rms_rel < 1e-2;
  }
  return ok;
}

bool TestWprojIdentity() {
  // Complex path with zero imaginary weights must equal the real path to
  // half-precision rounding (32-bit accumulation; run-to-run noise shown for scale)
  auto cfg = SmallConfig();
  cfg.accum16 = false;
  ImagerHarness h(cfg);
  std::vector<Source> srcs = {Dir(0, 90, 2.0, true), Dir(120, 40, 2.0, true)};
  h.SimulateSources(srcs, 0.5);
  h.UploadFEng();
  auto noise = RunToRun(h);
  h.LaunchGulp(true, false);
  h.Sync();
  auto ref = h.DownloadOut();
  h.SetComplexElements(RealAsComplex(h.GcfElements()));
  h.LaunchGulp(true, false);
  h.Sync();
  auto d = Compare(ref, h.DownloadOut());
  std::printf("    complex vs real: max %.2e, rms %.2e of peak (run-to-run: max %.2e, rms %.2e)\n",
              d.max_rel, d.rms_rel, noise.max_rel, noise.rms_rel);
  // The complex path multiplies phase x weight in float before the half-precision
  // voltage multiply, the real path scales after it, so rounding differs slightly.
  return d.rms_rel < 1e-3 && d.max_rel < 5e-2;
}

bool TestStreamsMatchSingleLaunch() {
  // ProcessGulp splits the channels over streams (chan_offset > 0). The image
  // of every channel must not depend on which stream processed it.
  auto cfg = SmallConfig();
  cfg.nchan = 32;
  cfg.nstreams = 8;
  cfg.accum16 = false;
  ImagerHarness h(cfg);
  std::vector<Source> srcs = {Dir(200, 35, 2.0, false), Dir(20, 60, 2.0, false)};
  h.SimulateSources(srcs, 0.5);
  h.UploadFEng();
  auto noise = RunToRun(h);
  h.LaunchAll(h.stream(0), true);  // one launch: blockIdx.x is the channel
  h.Sync();
  auto single = h.DownloadOut();
  h.LaunchGulp(true, false);       // 8 streams with chan_offset
  h.Sync();
  auto split = h.DownloadOut();
  auto d = Compare(single, split);
  // Per-channel peak flux ratio split/single for the source at el 35
  double worst = 0;
  for (int c = 0; c < cfg.nchan; ++c) {
    auto a = h.ImageXX(single, c), b = h.ImageXX(split, c);
    double ex, ey, px, py;
    ImagerHarness::ExpectedPixel(srcs[0].l, srcs[0].m, &ex, &ey);
    if (!FindPeak(a, ex, ey, 2, &px, &py)) continue;
    int ix = int(std::lround(px)), iy = int(std::lround(py));
    worst = std::max(worst, std::abs(1.0 - b[iy * ImagerHarness::N + ix] / a[iy * ImagerHarness::N + ix]));
  }
  std::printf("    streams vs single launch: max %.2e, rms %.2e of peak (run-to-run: max %.2e, rms %.2e); "
              "worst per-channel peak change %.1f%%\n",
              d.max_rel, d.rms_rel, noise.max_rel, noise.rms_rel, 100 * worst);
  return d.rms_rel <= 2 * noise.rms_rel + 1e-5 && d.max_rel <= 2 * noise.max_rel + 1e-4;
}

// Peak position (px) and peak value of each source, averaged over channels
struct SourceStats {
  std::vector<double> x, y, peak;
};
SourceStats MeasureSources(const ImagerHarness& h, const std::vector<float>& out,
                           const std::vector<Source>& srcs) {
  SourceStats st;
  for (const auto& s : srcs) {
    double sx = 0, sy = 0, sp = 0;
    int n = 0;
    for (int c = 0; c < h.config().nchan; ++c) {
      auto img = h.ImageXX(out, c);
      double ex, ey, px, py;
      ImagerHarness::ExpectedPixel(s.l, s.m, &ex, &ey);
      if (!FindPeak(img, ex, ey, 3, &px, &py)) continue;
      sx += px; sy += py;
      sp += img[int(std::lround(py)) * ImagerHarness::N + int(std::lround(px))];
      ++n;
    }
    st.x.push_back(sx / n); st.y.push_back(sy / n); st.peak.push_back(sp / n);
  }
  return st;
}

bool TestWeightModesAgree() {
  // Every GcfWeightMode must image the sources like the production (global)
  // mode: same peak positions (< 0.05 px) and peak values (< 1%), for the real
  // and the w-projected kernels. Pixel-level differences are also printed; they
  // include the reordering of half-precision atomics between kernel builds.
  bool ok = true;
  std::vector<Source> srcs = {Dir(124, 35, 2.0, true), Dir(20, 70, 2.0, true), Dir(250, 50, 2.0, true)};
  for (bool wproj : {false, true}) {
    std::vector<float> ref;
    SourceStats ref_st;
    for (int m = kGcfGlobal; m <= kGcfRegHalf + 1; ++m) {
      auto cfg = SmallConfig();
      cfg.accum16 = false;
      cfg.pc = PhaseCenter::kZenith;
      cfg.weight_mode = m <= kGcfRegHalf ? m : kProduction;  // last: GetImagingKernel
      ImagerHarness h(cfg);
      h.SimulateSources(srcs, 0.5);
      h.UploadFEng();
      if (wproj) h.SetComplexElements(WprojElements(h, h.GcfElements()));
      h.LaunchGulp(true, false);
      h.Sync();
      auto out = h.DownloadOut();
      auto st = MeasureSources(h, out, srcs);
      if (m == kGcfGlobal) {
        ref = out;
        ref_st = st;
        auto d = RunToRun(h);
        std::printf("    %-11s global: run-to-run pixels max %.2e, rms %.2e of peak\n",
                    wproj ? "w-projected" : "real", d.max_rel, d.rms_rel);
        continue;
      }
      double dpos = 0, dpeak = 0;
      for (size_t i = 0; i < srcs.size(); ++i) {
        dpos = std::max(dpos, std::hypot(st.x[i] - ref_st.x[i], st.y[i] - ref_st.y[i]));
        dpeak = std::max(dpeak, std::abs(st.peak[i] / ref_st.peak[i] - 1));
      }
      auto d = Compare(ref, out);
      bool pass = dpos < 0.05 && dpeak < 0.01;
      std::printf("    %-11s %-16s vs global: position %.3f px, peak %.2f%%; pixels max %.2e, rms %.2e %s\n",
                  wproj ? "w-projected" : "real", WeightModeName(m), dpos, 100 * dpeak, d.max_rel, d.rms_rel,
                  pass ? "" : "<-- FAIL");
      ok &= pass;
    }
  }
  return ok;
}

bool TestFluxVsElevation() {
  // Equal point sources placed exactly on pixel centres from zenith to 20 deg
  // elevation. After the gridding correction their peaks must be equal; the
  // previous (legacy) correction leaves them fainter towards the horizon.
  const double dl = pixel_dl();
  struct Pt { int dx, dy; };
  std::vector<Pt> pts;
  std::vector<double> els;
  for (double el : {90.0, 60.0, 45.0, 30.0, 20.0}) {
    int r = int(std::lround(std::cos(el * M_PI / 180) / dl));
    for (int q = 0; q < (el == 90 ? 1 : 2); ++q) {  // two azimuths per ring
      pts.push_back(q == 0 ? Pt{r, 0} : Pt{0, -r});
      els.push_back(el);
    }
  }
  std::vector<Source> srcs;
  for (auto p : pts) {
    // pixel x = N/2 - l/dl, y = N/2 - 1 + m/dl
    srcs.push_back({-p.dx * dl, p.dy * dl, 1.0, false});
  }
  bool ok = true;
  for (int corr : {kCorrUnit, kCorrLegacy, kCorrExact}) {
    for (bool wproj : {false, true}) {
      if (wproj && corr != kCorrExact) continue;
      auto cfg = SmallConfig();
      cfg.accum16 = false;
      cfg.correction = corr;
      ImagerHarness h(cfg);
      h.SimulateSources(srcs, 0.3);
      h.UploadFEng();
      if (wproj) h.SetComplexElements(WprojElements(h, h.GcfElements()));
      h.LaunchGulp(true, false);
      h.Sync();
      auto out = h.DownloadOut();
      std::vector<double> peak(srcs.size(), 0.0);
      for (int c = 0; c < cfg.nchan; ++c) {
        auto img = h.ImageXX(out, c);
        for (size_t i = 0; i < srcs.size(); ++i) {
          double ex, ey;
          ImagerHarness::ExpectedPixel(srcs[i].l, srcs[i].m, &ex, &ey);
          peak[i] += img[int(std::lround(ey)) * ImagerHarness::N + int(std::lround(ex))];
        }
      }
      const char* name = corr == kCorrUnit ? "no correction" : corr == kCorrLegacy ? "legacy correction" : "exact correction";
      std::printf("    %-17s %-11s peak / zenith peak:", name, wproj ? "w-projected" : "");
      double worst = 0;
      for (double el : {60.0, 45.0, 30.0, 20.0}) {
        double m = 0; int n = 0;
        for (size_t i = 0; i < srcs.size(); ++i) if (els[i] == el) { m += peak[i] / peak[0]; ++n; }
        m /= n;
        worst = std::max(worst, std::abs(m - 1));
        std::printf("  el%2.0f %.3f", el, m);
      }
      std::printf("\n");
      if (corr == kCorrExact) ok &= worst < 0.05;
    }
  }
  return ok;
}

bool TestWprojCorrectsW() {
  // Zenith phasing leaves the full stand heights (up to 1.9 m) as w
  auto cfg = SmallConfig();
  cfg.pc = PhaseCenter::kZenith;
  ImagerHarness h(cfg);
  std::vector<Source> srcs = {Dir(124, 35, 2.0, true), Dir(304, 35, 2.0, true),
                              Dir(34, 30, 2.0, true), Dir(214, 40, 2.0, true)};
  h.SimulateSources(srcs, 0.5);
  h.UploadFEng();
  h.LaunchGulp(true, false);
  h.Sync();
  auto before = MeasureOffsets(h, srcs);
  h.SetComplexElements(WprojElements(h, h.GcfElements()));
  h.LaunchGulp(true, false);
  h.Sync();
  auto after = MeasureOffsets(h, srcs);
  std::printf("    real kernels: mean %.3f px (max %.3f); w-projected: mean %.3f px (max %.3f)\n",
              before.mean, before.max, after.mean, after.max);
  return before.mean > 0.3 && after.mean < 0.5 * before.mean && after.mean < 0.2;
}

bool TestGulpAccumulation() {
  // A second gulp with is_first_gulp=false adds to the image (32-bit accumulation)
  auto cfg = SmallConfig();
  cfg.accum16 = false;
  ImagerHarness h(cfg);
  h.SimulateSources({Dir(60, 70, 2.0, false)}, 0.5);
  h.UploadFEng();
  auto noise = RunToRun(h);
  h.LaunchGulp(true, false);
  h.Sync();
  auto one = h.DownloadOut();
  h.LaunchGulp(false, false);
  h.Sync();
  auto d = Compare(one, h.DownloadOut(), 2.0);
  std::printf("    |(gulp1 + gulp2) - 2 gulp1|: max %.2e, rms %.2e of peak (run-to-run: max %.2e, rms %.2e)\n",
              d.max_rel, d.rms_rel, noise.max_rel, noise.rms_rel);
  return d.rms_rel <= 2 * noise.rms_rel + 1e-5 && d.max_rel <= 2 * noise.max_rel + 1e-4;
}



// Largest |value| of one output pol over all channels and pixels
double MaxAbsPol(const std::vector<float>& out, int pol) {
  double m = 0;
  for (size_t i = pol; i < out.size(); i += 4) m = std::max(m, double(std::abs(out[i])));
  return m;
}

bool TestAutocorrSingleStand() {
  // Each channel images one stand (all other gains zero), so the image is
  // P |g|^2 |K^(l, m)|^2 exactly and removal must leave only rounding. Y is X
  // rotated by 0.6 rad, so XY* = |X|^2 e^{-0.6i} and U and V both carry an
  // auto-correlation. Different stands per channel cover different sub-cell
  // offsets.
  const int stands[] = {0, 17, 42, 77, 101, 150, 199, 230};
  bool ok = true;
  for (bool wproj : {false, true}) {
    std::vector<float> out_off, out_on;
    for (bool remove : {false, true}) {
      auto cfg = SmallConfig();
      cfg.accum16 = false;
      cfg.correction = kCorrExact;
      cfg.remove_autocorr = remove;
      ImagerHarness h(cfg);
      std::vector<float> ph(size_t(cfg.nchan) * LWA_SV_NSTANDS * 4, 0.f);
      for (int c = 0; c < cfg.nchan; ++c) {
        float* g = &ph[(size_t(c) * LWA_SV_NSTANDS + stands[c % 8]) * 4];
        const float ax = 0.8f, px = 0.3f + 0.1f * c, ay = 1.3f, py = -1.1f;
        g[0] = ax * std::cos(px); g[1] = ax * std::sin(px);
        g[2] = ay * std::cos(py); g[3] = ay * std::sin(py);
      }
      h.SetPhases(ph);
      std::mt19937 rng(7);
      std::uniform_int_distribution<int> u(-5, 5);
      const double c6 = std::cos(0.6), s6 = std::sin(0.6);
      CNib2* v = reinterpret_cast<CNib2*>(h.host_f_eng());
      for (size_t i = 0; i < h.f_eng_bytes() / sizeof(CNib2); ++i) {
        int re = u(rng), im = u(rng);
        v[i].X.re = re; v[i].X.im = im;
        v[i].Y.re = int(std::lround(re * c6 - im * s6));
        v[i].Y.im = int(std::lround(re * s6 + im * c6));
      }
      h.UploadFEng();
      if (wproj) h.SetComplexElements(WprojElements(h, h.GcfElements()));
      h.LaunchGulp(true, false);
      h.Sync();
      (remove ? out_on : out_off) = h.DownloadOut();
    }
    std::printf("    %-11s residual max / auto-correlation peak:", wproj ? "w-projected" : "real");
    const char* names[] = {"XX", "YY", "U", "V"};
    for (int pol : {0, 1, 2, 3}) {
      double r = MaxAbsPol(out_on, pol) / MaxAbsPol(out_off, pol);
      std::printf("  %s %.2e", names[pol], r);
      ok &= r < 1e-2;
    }
    std::printf("\n");
  }
  return ok;
}

bool TestAutocorrPedestal() {
  // Sources plus noise on all stands. Without removal the image sits on the
  // total antenna power; with it the background (median over the sky) is ~0
  // and the source peaks above the background agree.
  std::vector<Source> srcs = {Dir(0, 90, 1.0, true), Dir(120, 45, 1.0, true),
                              Dir(250, 30, 1.0, true)};
  std::vector<double> level(2), rel_peak(2 * srcs.size());
  for (bool remove : {false, true}) {
    auto cfg = SmallConfig();
    cfg.accum16 = true;  // production accumulation
    cfg.correction = kCorrExact;
    cfg.remove_autocorr = remove;
    ImagerHarness h(cfg);
    // Noise-dominated voltages, as on the real sky: the auto-correlations are
    // ~(noise + source power) / (stands x source power) of the zenith peak
    h.SimulateSources(srcs, 4.0);
    h.UploadFEng();
    h.LaunchGulp(true, false);
    h.Sync();
    auto out = h.DownloadOut();
    std::vector<double> img(size_t(ImagerHarness::N) * ImagerHarness::N, 0.0);
    for (int c = 0; c < cfg.nchan; ++c) {
      auto im = h.ImageXX(out, c);
      for (size_t i = 0; i < img.size(); ++i) img[i] += im[i];
    }
    // sky pixels: within 60 deg of zenith
    const double dl = pixel_dl(), r_max = std::sin(60 * M_PI / 180) / dl;
    std::vector<double> sky;
    for (int y = 0; y < ImagerHarness::N; ++y)
      for (int x = 0; x < ImagerHarness::N; ++x)
        if (std::hypot(x - ImagerHarness::N / 2, y - (ImagerHarness::N / 2 - 1)) < r_max)
          sky.push_back(img[size_t(y) * ImagerHarness::N + x]);
    std::nth_element(sky.begin(), sky.begin() + sky.size() / 2, sky.end());
    const double med = sky[sky.size() / 2];
    double ex, ey;
    ImagerHarness::ExpectedPixel(srcs[0].l, srcs[0].m, &ex, &ey);
    const double zen = img[size_t(std::lround(ey)) * ImagerHarness::N + std::lround(ex)];
    level[remove] = med / zen;
    for (size_t s = 0; s < srcs.size(); ++s) {
      ImagerHarness::ExpectedPixel(srcs[s].l, srcs[s].m, &ex, &ey);
      double pk = img[size_t(std::lround(ey)) * ImagerHarness::N + std::lround(ex)];
      rel_peak[remove * srcs.size() + s] = (pk - med) / (zen - med);
    }
  }
  std::printf("    median sky level / zenith peak: %.3f without removal, %.4f with\n", level[0], level[1]);
  double worst = 0;
  for (size_t s = 1; s < srcs.size(); ++s) {
    std::printf("    source %zu peak above background / zenith: %.3f without, %.3f with\n", s,
                rel_peak[s], rel_peak[srcs.size() + s]);
    worst = std::max(worst, std::abs(rel_peak[srcs.size() + s] - rel_peak[s]));
  }
  // Without auto-correlations the image has zero mean over the whole grid, so
  // the sky median is slightly negative (WSClean: -0.029 on the 41-source
  // simulation). Peaks above a global median shift by up to ~1.5% because the
  // removed term rises towards the horizon after the gridding correction.
  return level[0] > 0.03 && std::abs(level[1]) < 0.02 && worst < 0.03;
}

bool TestStokesU() {
  // X and Y receive the same source signal (plus independent noise), so at the
  // source XY* = |S|^2: U = Re(XY*) equals XX there and V = Im(XY*) is ~0.
  auto cfg = SmallConfig();
  cfg.accum16 = false;
  cfg.correction = kCorrExact;
  cfg.remove_autocorr = true;
  ImagerHarness h(cfg);
  std::vector<Source> srcs = {Dir(0, 90, 2.0, true), Dir(120, 50, 2.0, true)};
  h.SimulateSources(srcs, 0.3);
  h.UploadFEng();
  h.LaunchGulp(true, false);
  h.Sync();
  auto out = h.DownloadOut();
  bool ok = true;
  for (size_t s = 0; s < srcs.size(); ++s) {
    double ex, ey, xx = 0, u = 0, v = 0;
    ImagerHarness::ExpectedPixel(srcs[s].l, srcs[s].m, &ex, &ey);
    for (int c = 0; c < cfg.nchan; ++c) {
      // image_file unpacking: pixel (x, y) is out[chan][a][b], a = (x + N/2) % N,
      // b = ((N - 1 - y) + N/2) % N
      const int N = ImagerHarness::N;
      const int x = int(std::lround(ex)), y = int(std::lround(ey));
      const size_t idx = (size_t(c) * N * N + size_t((x + N / 2) % N) * N + ((N - 1 - y) + N / 2) % N) * 4;
      xx += out[idx + 0];
      u += out[idx + 2];
      v += out[idx + 3];
    }
    std::printf("    source %zu: U / XX %.4f, V / XX %+.4f\n", s, u / xx, v / xx);
    ok &= std::abs(u / xx - 1) < 0.02 && std::abs(v / xx) < 0.02;
  }
  return ok;
}

// XX image summed over channels, and its value at the brightest pixel
std::vector<double> XXSum(const ImagerHarness& h, const std::vector<float>& out) {
  std::vector<double> img(size_t(ImagerHarness::N) * ImagerHarness::N, 0.0);
  for (int c = 0; c < h.config().nchan; ++c) {
    auto im = h.ImageXX(out, c);
    for (size_t i = 0; i < img.size(); ++i) img[i] += im[i];
  }
  return img;
}

bool TestChunkedLaunchMatches() {
  bool ok = true;
  for (bool wproj : {false, true}) {
    std::vector<float> ref;
    for (int chunk : {0, 37}) {
      auto cfg = SmallConfig();
      cfg.accum16 = false;
      cfg.correction = kCorrExact;
      cfg.remove_autocorr = true;
      cfg.seq_chunk = chunk;
      ImagerHarness h(cfg);
      h.SimulateSources({Dir(0, 90, 1.0, true), Dir(200, 40, 1.0, true)}, 2.0);
      h.UploadFEng();
      if (wproj) h.SetComplexElements(WprojElements(h, h.GcfElements()));
      h.LaunchGulp(true, false);
      h.Sync();
      if (chunk == 0) {
        ref = h.DownloadOut();
        continue;
      }
      auto d = Compare(ref, h.DownloadOut());
      std::printf("    %-11s launches of 37 vs one launch: max %.2e, rms %.2e of peak\n",
                  wproj ? "w-projected" : "real", d.max_rel, d.rms_rel);
      ok &= d.max_rel < 1e-3 && d.rms_rel < 1e-5;
    }
  }
  return ok;
}

bool TestAccum16Chunked() {
  // Production gulp length; the stall only shows over long runs
  std::vector<std::vector<double>> imgs;
  for (int mode : {0, 1, 2}) {  // 32-bit; 16-bit one launch; 16-bit launches of 100
    auto cfg = SmallConfig();
    cfg.nseq = 1000;
    cfg.nchan = 4;
    cfg.nstreams = 1;
    cfg.correction = kCorrExact;
    cfg.remove_autocorr = true;
    cfg.accum16 = mode != 0;
    cfg.seq_chunk = mode == 2 ? 100 : 0;
    ImagerHarness h(cfg);
    h.SimulateSources({Dir(0, 90, 1.0, true), Dir(60, 45, 1.0, true)}, 2.0);
    h.UploadFEng();
    h.LaunchGulp(true, false);
    h.Sync();
    imgs.push_back(XXSum(h, h.DownloadOut()));
  }
  const auto& ref = imgs[0];
  const size_t ipk = std::max_element(ref.begin(), ref.end()) - ref.begin();
  double r[3], rms[3];
  for (int m = 1; m < 3; ++m) {
    r[m] = imgs[m][ipk] / ref[ipk];
    double ss = 0;
    for (size_t i = 0; i < ref.size(); ++i) ss += std::pow((imgs[m][i] - ref[i]) / ref[ipk], 2);
    rms[m] = std::sqrt(ss / ref.size());
  }
  std::printf("    16-bit / 32-bit at the peak: one launch %.3f (rms %.1e of peak), launches of 100 %.4f (rms %.1e)\n",
              r[1], rms[1], r[2], rms[2]);
  return std::abs(r[2] - 1) < 0.01 && rms[2] < 5e-3;
}

}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "--quick") == 0) g_quick = true;
  int dev;
  cudaDeviceProp prop;
  EPIC_CHECK(cudaGetDevice(&dev));
  EPIC_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::printf("GPU %d: %s (sm_%d%d)\n", dev, prop.name, prop.major, prop.minor);

  struct T { const char* name; std::function<bool()> fn; };
  std::vector<T> tests = {{"gcf_normalised", TestGcfNormalised},
                          {"point_positions", TestPointPositions},
                          {"run_to_run_noise", TestRunToRunNoise},
                          {"streams_match_single_launch", TestStreamsMatchSingleLaunch},
                          {"wproj_identity", TestWprojIdentity},
                          {"weight_modes_agree", TestWeightModesAgree},
                          {"flux_vs_elevation", TestFluxVsElevation},
                          {"wproj_corrects_w", TestWprojCorrectsW},
                          {"gulp_accumulation", TestGulpAccumulation},
                          {"acorr_single_stand", TestAutocorrSingleStand},
                          {"acorr_pedestal", TestAutocorrPedestal},
                          {"stokes_u", TestStokesU},
                          {"chunked_launch_matches", TestChunkedLaunchMatches},
                          {"accum16_chunked", TestAccum16Chunked}};
  int failed = 0;
  for (auto& t : tests) {
    std::printf("[ RUN  ] %s\n", t.name);
    bool ok = false;
    try {
      ok = t.fn();
    } catch (const std::exception& e) {
      std::printf("    exception: %s\n", e.what());
    }
    std::printf("[ %s ] %s\n", ok ? "PASS" : "FAIL", t.name);
    failed += !ok;
  }
  std::printf("%d/%zu tests passed\n", int(tests.size()) - failed, tests.size());
  return failed ? 1 : 0;
}
