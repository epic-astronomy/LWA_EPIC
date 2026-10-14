// Gulp time budget check for the epic++ imaging kernel.
//
// Runs gulps exactly as MOFFCuHandler::ProcessGulp does (per-stream H2D copy
// of the F-engine data from pinned memory, one launch per stream, and on the
// last gulp of each --seq-accum-ms image the D2H copy of the image) and fails
// if the 99th-percentile time per image exceeds the data it covers: an image
// of 40 ms of data must be processed in under 40 ms for the pipeline to keep
// up. With --nseq below 1000 an image is several gulps.
//
// usage: ./gulp_budget [--gulps 200] [--seq-accum-ms 40] [--wproj 0|1|both]
//                      [--support 5] [--nseq 1000] [--nchan 128] [--nstreams 8]
//                      [--weight-mode -1..4]  (-1 = GetImagingKernel as in production,
//                       0..4 = GcfWeightMode) [--remove-autocorr 0|1|both]
//                      [--accum16 0|1] [--seq-chunk N]  (sequences per launch, 0 = whole gulp)
//                      [--graph 0|1]  (replay each gulp as a CUDA graph)
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "epic_harness.cuh"

using namespace epic_test;

namespace {

struct Stats {
  double mean, p50, p99, max;
};

Stats Summarise(std::vector<double> ms) {
  std::sort(ms.begin(), ms.end());
  double sum = 0;
  for (double v : ms) sum += v;
  auto q = [&](double p) { return ms[std::min(ms.size() - 1, size_t(p * (ms.size() - 1) + 0.5))]; };
  return {sum / ms.size(), q(0.5), q(0.99), ms.back()};
}

struct Result {
  Stats gulp, image;
};

// Gulps as MOFFCuHandler::ProcessGulp and the correlator raft kernel run them:
// per stream an H2D copy and a launch; on the last gulp of each image
// (gulps_per_image) also the D2H copy of the image. ProcessGulp waits for the
// GPU at the end of every gulp, so an image takes the sum of its gulps.
Result Measure(const ImagerConfig& cfg, int ngulps, int gulps_per_image, bool graph) {
  ImagerHarness h(cfg);
  h.SimulateNoise();
  h.UploadFEng();
  if (cfg.wproj) h.SetComplexElements(WprojElements(h, h.GcfElements()));
  float* host_out = nullptr;
  EPIC_CHECK(cudaMallocHost(&host_out, h.out_floats() * sizeof(float)));
  cudaStream_t ts;
  EPIC_CHECK(cudaStreamCreate(&ts));
  cudaEvent_t t0, t1, sync_ev;
  std::vector<cudaEvent_t> done(cfg.nstreams);
  EPIC_CHECK(cudaEventCreate(&t0));
  EPIC_CHECK(cudaEventCreate(&t1));
  EPIC_CHECK(cudaEventCreateWithFlags(&sync_ev, cudaEventDisableTiming));
  for (auto& e : done) EPIC_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  std::vector<double> ms, img_ms;
  double img_acc = 0;
  const int warm = std::max(5, gulps_per_image);
  for (int g = -warm; g < ngulps; ++g) {
    const int k = (g + warm) % gulps_per_image;  // position in the image
    const bool first = k == 0, last = k == gulps_per_image - 1;
    EPIC_CHECK(cudaEventRecord(t0, ts));
    if (graph) {
      h.LaunchGulpGraph(ts, first, last, host_out);
    } else {
      EPIC_CHECK(cudaEventRecord(sync_ev, ts));
      for (int i = 0; i < cfg.nstreams; ++i) EPIC_CHECK(cudaStreamWaitEvent(h.stream(i), sync_ev, 0));
      h.LaunchGulp(first, true);
      if (last) h.CopyOutAsync(host_out);
      for (int i = 0; i < cfg.nstreams; ++i) {
        EPIC_CHECK(cudaEventRecord(done[i], h.stream(i)));
        EPIC_CHECK(cudaStreamWaitEvent(ts, done[i], 0));
      }
    }
    EPIC_CHECK(cudaEventRecord(t1, ts));
    EPIC_CHECK(cudaEventSynchronize(t1));
    float m;
    EPIC_CHECK(cudaEventElapsedTime(&m, t0, t1));
    if (g < 0) continue;
    ms.push_back(m);
    img_acc += m;
    if (last) {
      img_ms.push_back(img_acc);
      img_acc = 0;
    }
  }
  EPIC_CHECK(cudaGetLastError());
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  cudaEventDestroy(sync_ev);
  for (auto& e : done) cudaEventDestroy(e);
  cudaStreamDestroy(ts);
  cudaFreeHost(host_out);
  return {Summarise(ms), img_ms.empty() ? Stats{0, 0, 0, 0} : Summarise(img_ms)};
}

}  // namespace

int main(int argc, char** argv) {
  int ngulps = 200;
  double seq_accum = 40.0;  // production --seq_accum (ms per image)
  std::string wproj = "both", acorr = "0";
  bool graph = false;
  ImagerConfig base;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--gulps") ngulps = std::stoi(v);
    else if (k == "--seq-accum-ms") seq_accum = std::stod(v);
    else if (k == "--wproj") wproj = v;
    else if (k == "--support") base.support = std::stoi(v);
    else if (k == "--nseq") base.nseq = std::stoi(v);
    else if (k == "--nchan") base.nchan = std::stoi(v);
    else if (k == "--nstreams") base.nstreams = std::stoi(v);
    else if (k == "--weight-mode") base.weight_mode = std::stoi(v);
    else if (k == "--remove-autocorr") acorr = v;
    else if (k == "--accum16") base.accum16 = std::stoi(v) != 0;
    else if (k == "--seq-chunk") base.seq_chunk = std::stoi(v);
    else if (k == "--graph") graph = std::stoi(v) != 0;
    else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
  }
  // One sequence is 40 us of data (25 kHz channels)
  const double gulp_ms = base.nseq * 0.04;
  const int gulps_per_image = std::max(1, int(std::lround(seq_accum / gulp_ms)));
  int dev;
  cudaDeviceProp prop;
  EPIC_CHECK(cudaGetDevice(&dev));
  EPIC_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::printf("GPU %d: %s (sm_%d%d); support %d, nseq %d (%.2f ms of data), nchan %d, nstreams %d, "
              "weights %s, accum %s, %d sequences per launch%s, %d gulps per %.0f ms image, %d gulps\n",
              dev, prop.name, prop.major, prop.minor, base.support, base.nseq, gulp_ms, base.nchan,
              base.nstreams, WeightModeName(base.weight_mode), base.accum16 ? "16-bit" : "32-bit",
              base.seq_chunk > 0 ? base.seq_chunk : base.nseq, graph ? " (CUDA graph)" : "", gulps_per_image, gulps_per_image * gulp_ms, ngulps);
  std::vector<bool> modes, acorrs;
  if (wproj == "0" || wproj == "both") modes.push_back(false);
  if (wproj == "1" || wproj == "both") modes.push_back(true);
  if (acorr == "0" || acorr == "both") acorrs.push_back(false);
  if (acorr == "1" || acorr == "both") acorrs.push_back(true);
  bool ok = true;
  std::printf("%-6s %-8s %10s %10s %10s %10s %8s %8s\n", "wproj", "autocorr", "gulp_mean", "gulp_p99",
              "image_mean", "image_p99", "load", "result");
  for (bool a : acorrs) {
    for (bool w : modes) {
      auto cfg = base;
      cfg.wproj = w;
      cfg.remove_autocorr = a;
      auto r = Measure(cfg, ngulps, gulps_per_image, graph);
      // Real time: each image (its gulps, then the D2H copy) must take less
      // than the data it covers
      const double budget = gulps_per_image * gulp_ms;
      bool pass = r.image.p99 < budget;
      ok &= pass;
      std::printf("%-6s %-8s %10.3f %10.3f %10.3f %10.3f %7.1f%% %8s\n", w ? "on" : "off",
                  a ? "removed" : "kept", r.gulp.mean, r.gulp.p99, r.image.mean, r.image.p99,
                  100.0 * r.image.mean / budget, pass ? "PASS" : "OVER");
    }
  }
  return ok ? 0 : 1;
}
