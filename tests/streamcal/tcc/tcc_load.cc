// Tensor-Core Correlator load generator for GPU-sharing tests with epic++.
//
//   tcc_load SECONDS PERIOD_MS [NCHAN NRECV NSAMP [COPY]]
// COPY = 1 copies the block from pinned host memory before every correlation (PCIe load of a
// real tap: 65.5 MB per gulp).
// Correlates one gulp-sized block (default 128 channels x 1024 samples x 256 stands x 2 pols,
// 4-bit) per PERIOD_MS (0: back to back) for SECONDS, accumulating on the GPU, and reports
// the number of correlations and the latency (launch to completion) distribution.
#include <cudawrappers/cu.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "libtcc/Correlator.h"

int main(int argc, char** argv) {
  const double seconds = argc > 1 ? std::atof(argv[1]) : 30;
  const double period_ms = argc > 2 ? std::atof(argv[2]) : 41.7959;
  const unsigned nchan = argc > 3 ? std::atoi(argv[3]) : 128;
  const unsigned nrecv = argc > 4 ? std::atoi(argv[4]) : 256;
  const unsigned nsamp = argc > 5 ? std::atoi(argv[5]) : 1024;
  const bool copy = argc > 6 && std::atoi(argv[6]) != 0;
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
  tcc::Correlator correlator(device, tcc::i4, nrecv, nchan, nsamp);
  cu::Stream stream;
  const size_t sbytes = size_t(nchan) * nsamp * nrecv * 2;
  const size_t vbytes = size_t(nchan) * nrecv * (nrecv + 1) / 2 * 4 * 8;
  cu::DeviceMemory d_s(sbytes), d_v(vbytes);
  cu::HostMemory h(sbytes);
  unsigned char* hp = static_cast<unsigned char*>(static_cast<void*>(h));
  for (size_t i = 0; i < sbytes; ++i) hp[i] = static_cast<unsigned char>((i * 2654435761u >> 13) & 0x77);
  stream.memcpyHtoDAsync(d_s, h, sbytes);
  stream.synchronize();
  using clk = std::chrono::steady_clock;
  std::vector<double> lat;
  auto t0 = clk::now();
  for (long k = 0;; ++k) {
    auto due = t0 + std::chrono::duration_cast<clk::duration>(std::chrono::duration<double, std::milli>(k * period_ms));
    if (period_ms > 0) std::this_thread::sleep_until(due);
    if (std::chrono::duration<double>(clk::now() - t0).count() > seconds) break;
    auto a = clk::now();
    if (copy) stream.memcpyHtoDAsync(d_s, h, sbytes);
    correlator.launchAsync(stream, d_v, d_s, k > 0);
    stream.synchronize();
    lat.push_back(std::chrono::duration<double, std::milli>(clk::now() - a).count());
  }
  double el = std::chrono::duration<double>(clk::now() - t0).count();
  std::sort(lat.begin(), lat.end());
  auto q = [&](double p) { return lat[std::min(lat.size() - 1, size_t(p * lat.size()))]; };
  std::printf("tcc_load%s: %zu correlations in %.1f s (%.1f per s; one per gulp needs 23.9 per s); latency ms "
              "p50 %.3f p90 %.3f p99 %.3f max %.3f\n", copy ? " (with host-to-device copy)" : "", lat.size(), el, lat.size() / el, q(0.5), q(0.9), q(0.99), lat.back());
  return 0;
}
