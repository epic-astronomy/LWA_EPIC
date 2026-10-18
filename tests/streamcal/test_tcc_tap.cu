// Check the GPU calibration tap (src/ex/tcc_tap.cu) against the CPU X-engine
// (src/ex/xengine.hpp) on random gulps. Run on a GPU node; build with
// tests/streamcal/tcc/build.sh, which also builds this test.
//
//   test_tcc_tap <vis_path> [ngulps]
//
// Feeds `ngulps` random 128-channel gulps through the tap the way ProcessGulp
// does (BeforeCopy, copy into one device buffer, AfterCopy, Launch), with an
// integration of exactly `ngulps` gulps, then compares the published table
// with the X-engine's sums. Reusing a single device buffer also checks that a
// gulp is not overwritten before the tap has read it.
#include <cuda_runtime_api.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "constants.h"
#include "tcc_tap.h"
#include "vis_table.hpp"
#include "xengine.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s VIS_PATH [NGULPS]\n", argv[0]);
    return 2;
  }
  const std::string path = argv[1];
  const int ngulps = argc > 2 ? std::atoi(argv[2]) : 6;
  constexpr int kNchan = 128, kNseq = 1000, kN = LWA_SV_NSTANDS, kP = LWA_SV_NPOLS;
  const size_t gulp_bytes = size_t(kNchan) * kNseq * LWA_SV_INP_PER_CHAN;
  unlink(path.c_str());

  TccTapOptions o;
  o.path = path;
  o.interval_s = float(ngulps * kNseq * SAMPLING_LEN_uS * 1e-6 * 0.999);
  TccTap tap(o, 0, kNchan, kNseq, 1);

  uint8_t *h_gulp, *d_gulp;
  cudaMallocHost(&h_gulp, gulp_bytes);
  cudaMalloc(&d_gulp, gulp_bytes);
  cudaStream_t s;
  cudaStreamCreate(&s);
  std::vector<int64_t> re(size_t(kNchan) * kP * kN * kN, 0), im(re.size(), 0);
  std::mt19937 rng(7);
  const uint64_t seq0 = 1000000;
  double t_launch = 0;
  for (int g = 0; g < ngulps; ++g) {
    // the host buffer is reused only after the previous copy has finished
    cudaStreamSynchronize(s);
    for (size_t i = 0; i < gulp_bytes; ++i) h_gulp[i] = uint8_t(rng());
    xengine::accumulate(h_gulp, kNseq, 0, kNchan, 0, kNseq, re.data(), im.data(), 16);
    auto t0 = std::chrono::steady_clock::now();
    if (!tap.Arm(seq0 + uint64_t(g) * kNseq, 3104, kNchan, kNseq, 1.f)) {
      std::printf("FAIL: gulp %d not armed\n", g);
      return 1;
    }
    tap.BeforeCopy(s);
    cudaMemcpyAsync(d_gulp, h_gulp, gulp_bytes, cudaMemcpyHostToDevice, s);
    tap.AfterCopy(0, s);
    tap.Launch(d_gulp);
    t_launch += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  // wait for the table
  const size_t bytes = sizeof(VisTableHeader) + size_t(kNchan) * kP * kN * kN * sizeof(std::complex<float>);
  std::vector<char> buf(bytes);
  bool got = false;
  for (int k = 0; k < 600 && !got; ++k) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::ifstream f(path, std::ios::binary);
    if (!f.read(buf.data(), bytes)) continue;
    auto* h = reinterpret_cast<VisTableHeader*>(buf.data());
    got = h->version >= 1 && (h->seq & 1) == 0;
  }
  if (!got) {
    std::printf("FAIL: no table published\n");
    return 1;
  }
  auto* h = reinterpret_cast<VisTableHeader*>(buf.data());
  auto* R = reinterpret_cast<std::complex<float>*>(buf.data() + sizeof(VisTableHeader));
  std::printf("table: version %llu, chan0 %lld, nchan %u, %u gulps, nsamp %llu, duty %.2f, note '%s'\n",
              (unsigned long long)h->version, (long long)h->chan0, h->nchan, h->ngulps,
              (unsigned long long)h->nsamp, h->duty, h->note);
  bool ok = h->nsamp == uint64_t(ngulps) * kNseq && h->ngulps == uint32_t(ngulps) && h->nchan == kNchan &&
            h->chan0 == 3104 && h->seq_first == seq0;
  const double inv = 1.0 / double(h->nsamp);
  double maxerr = 0, maxval = 0;
  for (size_t cp = 0; cp < size_t(kNchan) * kP; ++cp) {
    for (int a = 0; a < kN; ++a) {
      for (int b = 0; b < kN; ++b) {
        const size_t u = cp * kN * kN + size_t(std::min(a, b)) * kN + std::max(a, b);
        std::complex<double> ref(double(re[u]) * inv, double(im[u]) * inv);
        if (a > b) ref = std::conj(ref);
        const std::complex<double> got_ab(R[cp * kN * kN + size_t(a) * kN + b]);
        maxerr = std::max(maxerr, std::abs(got_ab - ref));
        maxval = std::max(maxval, std::abs(ref));
      }
    }
  }
  ok &= maxerr <= 1e-6 * maxval;
  std::printf("max |GPU tap - X-engine| = %.3g (largest |R| %.3g); host time per gulp in the tap calls %.3f ms\n",
              maxerr, maxval, t_launch / ngulps);
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  cudaFree(d_gulp);
  cudaFreeHost(h_gulp);
  return ok ? 0 : 1;
}
