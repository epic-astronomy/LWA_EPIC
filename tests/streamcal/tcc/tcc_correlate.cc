// Correlate a file of samples with the ASTRON Tensor-Core Correlator (libtcc).
//
//   tcc_correlate i4|i8 NRECV NCHAN NSAMP IN OUT [NBLOCK]
// IN  holds NBLOCK consecutive blocks of samples in TCC layout,
//     [chan][NSAMP / T][recv][pol 2][T] complex (i8: 2 bytes, T = 16; i4: 1 byte, T = 32),
//     which are correlated and accumulated on the GPU (launchAsync(..., add = true)).
// OUT receives complex<int32> visibilities [chan][baseline][polY][polX], with
//     baseline = recvX * (recvX + 1) / 2 + recvY, recvY <= recvX.
#include <cudawrappers/cu.hpp>

#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "libtcc/Correlator.h"

int main(int argc, char** argv) {
  if (argc < 7) {
    std::fprintf(stderr, "usage: %s i4|i8 NRECV NCHAN NSAMP IN OUT [NBLOCK]\n", argv[0]);
    return 2;
  }
  const std::string fmt = argv[1];
  const unsigned nrecv = std::atoi(argv[2]), nchan = std::atoi(argv[3]), nsamp = std::atoi(argv[4]);
  const unsigned nblock = argc > 7 ? std::atoi(argv[7]) : 1;
  const tcc::Format format = fmt == "i4" ? tcc::i4 : tcc::i8;
  const size_t sample_bytes = fmt == "i4" ? 1 : 2;
  const size_t block_bytes = size_t(nchan) * nsamp * nrecv * 2 * sample_bytes;
  const size_t nbl = size_t(nrecv) * (nrecv + 1) / 2;
  const size_t vis_bytes = size_t(nchan) * nbl * 4 * sizeof(std::complex<int32_t>);

  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
  tcc::Correlator correlator(device, format, nrecv, nchan, nsamp);
  cu::Stream stream;
  cu::DeviceMemory d_samples(block_bytes), d_vis(vis_bytes);
  cu::HostMemory h_samples(block_bytes), h_vis(vis_bytes);

  std::ifstream in(argv[5], std::ios::binary);
  auto t0 = std::chrono::steady_clock::now();
  double t_gpu = 0;
  for (unsigned b = 0; b < nblock; ++b) {
    if (!in.read(static_cast<char*>(static_cast<void*>(h_samples)), block_bytes)) {
      std::fprintf(stderr, "short read at block %u\n", b);
      return 1;
    }
    auto g0 = std::chrono::steady_clock::now();
    stream.memcpyHtoDAsync(d_samples, h_samples, block_bytes);
    correlator.launchAsync(stream, d_vis, d_samples, b > 0);
    stream.synchronize();
    t_gpu += std::chrono::duration<double>(std::chrono::steady_clock::now() - g0).count();
  }
  stream.memcpyDtoHAsync(h_vis, d_vis, vis_bytes);
  stream.synchronize();
  std::ofstream(argv[6], std::ios::binary).write(static_cast<const char*>(static_cast<void*>(h_vis)), vis_bytes);
  double t_all = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("tcc_correlate %s: %u recv x %u chan x %u samples x %u blocks; copy+correlate %.3f s, total %.3f s\n",
              fmt.c_str(), nrecv, nchan, nsamp, nblock, t_gpu, t_all);
  return 0;
}
