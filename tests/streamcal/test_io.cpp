// Tests for the calibration tap (cal_tap.hpp, xengine.hpp) and the gain table
// reader (gain_table.hpp). No CUDA needed:
//   g++ -O3 -march=znver3 -fopenmp -std=c++17 -I tests/streamcal/shim -I src/ex
//       tests/streamcal/test_io.cpp -o test_io -lpthread
//   ./test_io selftest <dir>                       C++-only checks
//   ./test_io tap <vis_path> <ngulps>              write a vis table from seeded data (Python checks it)
//   ./test_io readgain <gain_path> <chan0> <nchan> <seq>   print what the reader accepts
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "cal_tap.hpp"
#include "gain_table.hpp"

namespace {
constexpr int kNchan = 128, kGulpNseq = 1000, kTapNseq = 50;

std::vector<uint8_t> MakeGulp(uint32_t seed) {
  std::vector<uint8_t> g(size_t(kNchan) * kGulpNseq * LWA_SV_INP_PER_CHAN);
  std::mt19937 rng(seed);
  for (auto& v : g) v = uint8_t(rng());
  return g;
}

int RunTap(const std::string& path, int ngulps, std::vector<std::vector<uint8_t>>* keep = nullptr) {
  CalTapOptions o;
  o.path = path;
  o.nseq = kTapNseq;
  o.nthreads = 4;
  // publish after exactly ngulps gulps
  o.interval_s = float((ngulps - 0.5) * kGulpNseq * SAMPLING_LEN_uS * 1e-6);
  o.min_quality = 0.5f;
  CalTap tap(o);
  int offered = 0;
  for (int i = 0; i < ngulps; ++i) {
    auto g = MakeGulp(1000 + i);
    // retry until the worker is free (a real imager would skip instead)
    while (!tap.Offer(g.data(), uint64_t(i) * kGulpNseq, 3104, kNchan, kGulpNseq, 1.0f)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    tap.WaitCopied();
    ++offered;
    if (keep) keep->push_back(std::move(g));
  }
  while (tap.Version() < 1) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  return offered;
}

bool CheckVis(const std::string& path, const std::vector<std::vector<uint8_t>>& gulps) {
  int fd = ::open(path.c_str(), O_RDONLY);
  struct stat st;
  fstat(fd, &st);
  auto* base = static_cast<const uint8_t*>(mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0));
  ::close(fd);
  auto* h = reinterpret_cast<const VisTableHeader*>(base);
  auto* R = reinterpret_cast<const std::complex<float>*>(base + sizeof(VisTableHeader));
  bool ok = h->nsamp == uint64_t(kTapNseq) * gulps.size() && h->nchan == kNchan && (h->seq & 1) == 0;
  for (int c : {0, 63, 127})
    for (int p : {0, 1})
      for (int a : {0, 17, 255})
        for (int b : {0, 100, 255}) {
          std::complex<double> s = 0;
          for (const auto& g : gulps)
            for (int t = 0; t < kTapNseq; ++t) {
              auto va = g[(size_t(c) * kGulpNseq + t) * LWA_SV_INP_PER_CHAN + a * 2 + p];
              auto vb = g[(size_t(c) * kGulpNseq + t) * LWA_SV_INP_PER_CHAN + b * 2 + p];
              std::complex<double> xa(xengine::hi_nib(va), xengine::lo_nib(va));
              std::complex<double> xb(xengine::hi_nib(vb), xengine::lo_nib(vb));
              s += xa * std::conj(xb);
            }
          s /= double(h->nsamp);
          auto v = R[((size_t(c) * 2 + p) * 256 + a) * 256 + b];
          if (std::abs(std::complex<double>(v) - s) > 1e-5 * (1 + std::abs(s))) ok = false;
        }
  munmap(const_cast<uint8_t*>(base), st.st_size);
  return ok;
}

void WriteGain(const std::string& path, uint64_t version, uint64_t seq, int64_t chan0, uint32_t nchan,
               uint64_t valid_from, std::complex<float> fill) {
  size_t bytes = sizeof(GainTableHeader) + size_t(nchan) * 512 * sizeof(std::complex<float>);
  int fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
  ftruncate(fd, bytes);
  auto* base = static_cast<uint8_t*>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  ::close(fd);
  auto* h = reinterpret_cast<GainTableHeader*>(base);
  std::memcpy(h->magic, "EPICGAIN", 8);
  h->layout = 1;
  h->header_bytes = sizeof(GainTableHeader);
  h->seq = seq;
  h->version = version;
  h->chan0 = chan0;
  h->nchan = nchan;
  h->nstand = 256;
  h->npol = 2;
  h->flags = 1;
  h->valid_from_seq = valid_from;
  auto* d = reinterpret_cast<std::complex<float>*>(base + sizeof(GainTableHeader));
  for (size_t i = 0; i < size_t(nchan) * 512; ++i) d[i] = fill * float(1 + i % 7);
  munmap(base, bytes);
}

int SelfTest(const std::string& dir) {
  int fails = 0;
  auto check = [&](bool ok, const char* what) {
    std::printf("%-58s %s\n", what, ok ? "ok" : "FAIL");
    fails += !ok;
  };
  // tap
  std::vector<std::vector<uint8_t>> gulps;
  std::string vis = dir + "/test_vis";
  RunTap(vis, 3, &gulps);
  check(CheckVis(vis, gulps), "tap: published R matches scalar reference (3 gulps)");
  // gain table
  std::string gp = dir + "/test_gain";
  GainTableReader rd(gp);
  check(!rd.Poll(3104, 128, 0), "gain: missing file -> no table");
  WriteGain(gp, 1, 2, 3040, 192, 0, {0.5f, 0.25f});
  auto s = rd.Poll(3104, 128, 100);
  check(s && s->version == 1 && s->nchan == 128 &&
            std::abs(s->corr[0] - std::complex<float>(0.5f, 0.25f) * float(1 + (64 * 512) % 7)) < 1e-6,
        "gain: table covering a wider band is sliced at chan0");
  check(!rd.Poll(3104, 128, 100), "gain: same version is not returned twice");
  WriteGain(gp, 2, 3, 3040, 192, 0, {1, 0});
  check(!rd.Poll(3104, 128, 100), "gain: odd seqlock (write in progress) -> ignored");
  WriteGain(gp, 2, 4, 3040, 192, 500, {1, 0});
  check(!rd.Poll(3104, 128, 100), "gain: valid_from_seq in the future -> deferred");
  check(static_cast<bool>(rd.Poll(3104, 128, 600)), "gain: ... and applied once the seq is reached");
  WriteGain(gp, 3, 6, 3200, 64, 0, {1, 0});
  check(!rd.Poll(3104, 128, 600), "gain: table not covering the imaged channels -> rejected");
  WriteGain(gp, 4, 8, 3040, 192, 0, {NAN, 0});
  check(!rd.Poll(3104, 128, 600), "gain: NaN -> rejected");
  WriteGain(gp, 5, 10, 3040, 192, 0, {100, 0});
  check(!rd.Poll(3104, 128, 600), "gain: |c| above clamp -> rejected");
  WriteGain(gp, 6, 12, 3040, 192, 0, {8, 0});
  check(!rd.Poll(3104, 128, 600), "gain: unnormalised table (median |c| 32) -> rejected");
  WriteGain(gp, 7, 14, 3040, 192, 0, {0.5f, 0});
  check(static_cast<bool>(rd.Poll(3104, 128, 600)), "gain: normalised table after rejections -> applied");
  std::vector<float> nominal(128 * 512 * 2, 0.f), out(nominal.size());
  for (size_t i = 0; i < nominal.size() / 2; ++i) { nominal[2 * i] = 0.f; nominal[2 * i + 1] = 1.f; }  // i
  GainSnapshot snap;
  snap.valid = true;
  snap.nchan = 128;
  snap.corr.assign(128 * 512, {2.f, 0.f});
  snap.corr[5] = {0.f, 0.f};
  ApplyGainCorrections(nominal.data(), snap, out.data());
  check(out[0] == 0.f && out[1] == 2.f && out[10] == 0.f && out[11] == 0.f, "gain: apply multiplies (and zero flags)");
  return fails;
}
}  // namespace

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "selftest";
  if (mode == "selftest") {
    int f = SelfTest(argc > 2 ? argv[2] : "/dev/shm");
    std::printf("%s\n", f ? "FAILED" : "all passed");
    return f != 0;
  }
  if (mode == "tap") {
    RunTap(argv[2], std::atoi(argv[3]));
    std::printf("published %s\n", argv[2]);
    return 0;
  }
  if (mode == "readgain") {
    GainTableReader rd(argv[2]);
    auto s = rd.Poll(std::atoll(argv[3]), std::atoi(argv[4]), std::strtoull(argv[5], nullptr, 10));
    if (!s) {
      std::printf("none %s\n", rd.LastError().c_str());
      return 1;
    }
    double sum_re = 0, sum_im = 0;
    for (auto& c : s->corr) { sum_re += c.real(); sum_im += c.imag(); }
    std::printf("version %llu valid %d n %zu sum %.6f %.6f first %.6f %.6f\n", (unsigned long long)s->version,
                int(s->valid), s->corr.size(), sum_re, sum_im, s->corr[0].real(), s->corr[0].imag());
    return 0;
  }
  return 2;
}
