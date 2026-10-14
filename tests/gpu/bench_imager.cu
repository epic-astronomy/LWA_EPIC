// nvbench benchmarks for the epic++ imaging kernel (BlockEpicImager).
//
//   epic_imager_kernel  all channels of one gulp in a single launch, kernel time only
//   epic_gulp           MOFFCuHandler::ProcessGulp equivalent: per-stream H2D copy of
//                       the F-engine data (pinned host memory) plus one launch per
//                       stream; this is what has to stay under the 40 ms gulp
//
// Axes: Support, Wproj (0 = real kernel path, 1 = complex w-projected weights),
// Accum16 (1 = --accum_16bit, production), NSeq (--nts), NChan (--channels),
// NStreams (--nstreams), WeightMode (-1 = the kernel GetImagingKernel picks, as
// in production; GcfWeightMode 0 global, 1 global-half2, 2 shared,
// 4 registers-half2), RemoveAutocorr (1 = --remove_autocorr). Defaults are the
// production settings. Configurations
// above 45 ms per gulp (mode 3, and mode 4 with w-projection) are skipped.
//
// Example: ./bench_imager --json epic1.json
//          ./bench_imager -b epic_gulp -a Wproj=[0,1]
#include <nvbench/nvbench.cuh>

#include <memory>
#include <vector>

#include "epic_harness.cuh"

using namespace epic_test;

namespace {

// Configurations measured above 45 ms per 40 ms gulp on an RTX 4090
// (gulp_budget): float2 weights in registers (~66 ms) and w-projected half2
// weights in registers (~49 ms). They are skipped.
bool TooSlow(const ImagerConfig& c) {
  return c.weight_mode == kGcfReg || (c.weight_mode == kGcfRegHalf && c.wproj);
}

ImagerConfig FromState(nvbench::state& state) {
  ImagerConfig c;
  c.support = int(state.get_int64("Support"));
  c.wproj = state.get_int64("Wproj") != 0;
  c.accum16 = state.get_int64("Accum16") != 0;
  c.nseq = int(state.get_int64("NSeq"));
  c.nchan = int(state.get_int64("NChan"));
  c.nstreams = int(state.get_int64("NStreams"));
  c.weight_mode = int(state.get_int64("WeightMode"));
  c.remove_autocorr = state.get_int64("RemoveAutocorr") != 0;
  return c;
}

std::unique_ptr<ImagerHarness> MakeHarness(const ImagerConfig& cfg) {
  auto h = std::make_unique<ImagerHarness>(cfg);
  h->SimulateNoise();
  h->UploadFEng();
  if (cfg.wproj) h->SetComplexElements(WprojElements(*h, h->GcfElements()));
  return h;
}

void epic_imager_kernel(nvbench::state& state) {
  auto cfg = FromState(state);
  if (TooSlow(cfg)) {
    state.skip("above 45 ms per gulp; not benchmarked");
    return;
  }
  auto h = MakeHarness(cfg);
  state.add_global_memory_reads<uint8_t>(h->f_eng_bytes(), "F-eng");
  state.add_element_count(size_t(cfg.nchan) * cfg.nseq, "Spectra");
  state.exec([&](nvbench::launch& launch) { h->LaunchAll(launch.get_stream(), true); });
}

void epic_gulp(nvbench::state& state) {
  auto cfg = FromState(state);
  if (TooSlow(cfg)) {
    state.skip("above 45 ms per gulp; not benchmarked");
    return;
  }
  auto h = MakeHarness(cfg);
  state.add_global_memory_reads<uint8_t>(h->f_eng_bytes(), "F-eng");
  state.add_element_count(size_t(cfg.nchan) * cfg.nseq, "Spectra");
  // The gulp runs on the harness's own streams (as in production). Bracket it
  // with events on nvbench's launch stream so the measured GPU time covers
  // every stream's copy and kernel.
  cudaEvent_t start_ev;
  std::vector<cudaEvent_t> done(cfg.nstreams);
  EPIC_CHECK(cudaEventCreateWithFlags(&start_ev, cudaEventDisableTiming));
  for (auto& e : done) EPIC_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    cudaStream_t ls = launch.get_stream();
    timer.start();
    EPIC_CHECK(cudaEventRecord(start_ev, ls));
    for (int i = 0; i < cfg.nstreams; ++i) EPIC_CHECK(cudaStreamWaitEvent(h->stream(i), start_ev, 0));
    h->LaunchGulp(true, true);
    for (int i = 0; i < cfg.nstreams; ++i) {
      EPIC_CHECK(cudaEventRecord(done[i], h->stream(i)));
      EPIC_CHECK(cudaStreamWaitEvent(ls, done[i], 0));
    }
    timer.stop();
  });
  cudaEventDestroy(start_ev);
  for (auto& e : done) cudaEventDestroy(e);
}

}  // namespace

NVBENCH_BENCH(epic_imager_kernel)
    .add_int64_axis("Support", {5})
    .add_int64_axis("Wproj", {0, 1})
    .add_int64_axis("Accum16", {1})
    .add_int64_axis("NSeq", {1000})
    .add_int64_axis("NChan", {128})
    .add_int64_axis("NStreams", {8})
    .add_int64_axis("WeightMode", {kProduction, kGcfGlobal, kGcfGlobalHalf, kGcfShared, kGcfRegHalf})
    .add_int64_axis("RemoveAutocorr", {0, 1})
    .set_timeout(300);

NVBENCH_BENCH(epic_gulp)
    .add_int64_axis("Support", {5})
    .add_int64_axis("Wproj", {0, 1})
    .add_int64_axis("Accum16", {1})
    .add_int64_axis("NSeq", {1000})
    .add_int64_axis("NChan", {128})
    .add_int64_axis("NStreams", {8})
    .add_int64_axis("WeightMode", {kProduction, kGcfGlobal, kGcfGlobalHalf, kGcfShared, kGcfRegHalf})
    .add_int64_axis("RemoveAutocorr", {0, 1})
    .set_timeout(300);
