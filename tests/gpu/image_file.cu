// Image F-engine data from a file with the production imaging kernel.
//
// Input from tools/tbn_to_feng.py: <prefix>.feng (CHAN_MAJOR 4-bit voltages for
// all spectra), <prefix>.phases (calibration phases), <prefix>.json (chan0,
// nchan, nseq). The data are processed in production-sized gulps (--nts 1000),
// accumulating like epic++, with the kernel GetImagingKernel picks, the exact
// gridding correction and, with --wproj, w-projected kernels.
//
// Output: <out>.f32, float32 [pol XX,YY][chan][y][x] unpacked as in
// epic_utils.save_output (east left, north up).
//
// usage: ./image_file <prefix> <out> [--wproj] [--correction unit|legacy|exact] [--dump] [--remove-autocorr] [--accum32] [--nts 1000]
//                     [--seq-chunk N]
//
// --dump also writes <out>.gcf (float32 [chan][stand][S*S] kernel elements) and
// <out>.antpos (float32 [chan][stand][3] grid positions), for offline analysis.
#include <cstdio>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "epic_harness.cuh"

using namespace epic_test;

static int JsonInt(const std::string& js, const std::string& key) {
  std::smatch m;
  if (!std::regex_search(js, m, std::regex("\"" + key + "\":\\s*(-?[0-9]+)")))
    throw std::runtime_error("missing " + key);
  return std::stoi(m[1]);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <prefix> <out> [--wproj] [--correction unit|legacy|exact] [--dump] [--remove-autocorr] [--accum32] [--nts 1000]\n", argv[0]);
    return 2;
  }
  std::string prefix = argv[1], outp = argv[2];
  bool wproj = false, dump = false, acorr = false, accum32 = false;
  int corr = kCorrExact;
  int nts = 1000;  // production --nts
  int chunk = 0;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--wproj") wproj = true;
    else if (a == "--dump") dump = true;
    else if (a == "--remove-autocorr") acorr = true;
    else if (a == "--accum32") accum32 = true;
    else if (a == "--nts" && i + 1 < argc) nts = std::stoi(argv[++i]);
    else if (a == "--seq-chunk" && i + 1 < argc) chunk = std::stoi(argv[++i]);
    else if (a == "--correction" && i + 1 < argc) {
      std::string v = argv[++i];
      corr = v == "unit" ? kCorrUnit : v == "legacy" ? kCorrLegacy : kCorrExact;
    }
  }
  std::ifstream jf(prefix + ".json");
  std::stringstream ss;
  ss << jf.rdbuf();
  std::string js = ss.str();
  const int chan0 = JsonInt(js, "chan0"), nchan = JsonInt(js, "nchan"), nseq_all = JsonInt(js, "nseq");

  ImagerConfig cfg;
  cfg.nseq = nts;
  cfg.nchan = nchan;
  cfg.nstreams = nchan;  // one channel per stream (production: 16 per stream)
  cfg.chan0 = chan0;
  cfg.correction = corr;
  cfg.remove_autocorr = acorr;
  cfg.accum16 = !accum32;  // production: --accum_16bit
  cfg.seq_chunk = chunk;
  ImagerHarness h(cfg);

  std::vector<float> ph(size_t(nchan) * LWA_SV_NSTANDS * 4);
  std::ifstream pf(prefix + ".phases", std::ios::binary);
  pf.read(reinterpret_cast<char*>(ph.data()), ph.size() * sizeof(float));
  if (!pf) throw std::runtime_error("short phases file");
  h.SetPhases(ph);
  if (wproj) h.SetComplexElements(WprojElements(h, h.GcfElements()));

  std::vector<uint8_t> all(size_t(nchan) * nseq_all * LWA_SV_INP_PER_CHAN);
  std::ifstream ff(prefix + ".feng", std::ios::binary);
  ff.read(reinterpret_cast<char*>(all.data()), all.size());
  if (!ff) throw std::runtime_error("short feng file");

  const int ngulps = nseq_all / cfg.nseq;
  const size_t row = LWA_SV_INP_PER_CHAN;
  for (int g = 0; g < ngulps; ++g) {
    // Re-pack this gulp's spectra channel-major, as the F-engine buffer would be
    for (int c = 0; c < nchan; ++c)
      std::copy(all.begin() + (size_t(c) * nseq_all + size_t(g) * cfg.nseq) * row,
                all.begin() + (size_t(c) * nseq_all + size_t(g + 1) * cfg.nseq) * row,
                h.host_f_eng() + size_t(c) * cfg.nseq * row);
    h.LaunchGulp(g == 0, true);
    h.Sync();
  }
  auto out = h.DownloadOut();
  // XX and YY images, unpacked like save_output
  std::vector<float> img(size_t(2) * nchan * ImagerHarness::N * ImagerHarness::N);
  const int N = ImagerHarness::N;
  for (int pol = 0; pol < 2; ++pol)
    for (int c = 0; c < nchan; ++c)
      for (int y = 0; y < N; ++y) {
        int b = ((N - 1 - y) + N / 2) % N;
        for (int x = 0; x < N; ++x) {
          int a = (x + N / 2) % N;
          img[((size_t(pol) * nchan + c) * N + y) * N + x] = out[(size_t(c) * N * N + size_t(a) * N + b) * 4 + pol];
        }
      }
  std::ofstream of(outp + ".f32", std::ios::binary);
  of.write(reinterpret_cast<const char*>(img.data()), img.size() * sizeof(float));
  if (dump) {
    auto g = h.GcfElements();
    std::ofstream(outp + ".gcf", std::ios::binary).write(reinterpret_cast<const char*>(g.data()), g.size() * sizeof(float));
    const auto& ap = h.host_antpos();
    std::ofstream(outp + ".antpos", std::ios::binary).write(reinterpret_cast<const char*>(ap.data()), ap.size() * sizeof(float));
  }
  if (acorr) {
    auto ak = h.AcorrKernels();
    double sum = 0;
    for (auto v : ak) sum += v.x;
    std::printf("auto-correlation removal on: %zu weight auto-correlation cells, sum of real parts %.3f\n", ak.size(), sum);
  }
  std::printf("imaged %d gulps x %d spectra, %d channels from %d; wproj %s, correction %s -> %s.f32\n", ngulps,
              cfg.nseq, nchan, chan0, wproj ? "on" : "off",
              corr == kCorrUnit ? "unit" : corr == kCorrLegacy ? "legacy" : "exact", outp.c_str());
  return 0;
}
