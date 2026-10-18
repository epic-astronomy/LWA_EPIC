// Driver for test_rfi_stage.py: reads a projector table (ProjTableReader) and a gulp file, runs
// RfiStage on the GPU and writes the processed gulp and the impulse count.
//   rfi_stage_driver PROJ GULP OUT NCHAN NSEQ NSTREAMS SIGMA
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>

#include "../../../src/ex/proj_table.hpp"
#include "../../../src/ex/rfi_stage.h"

int main(int argc, char** argv) {
  if (argc != 8) {
    std::fprintf(stderr, "usage: %s PROJ GULP OUT NCHAN NSEQ NSTREAMS SIGMA\n", argv[0]);
    return 2;
  }
  const int nchan = std::atoi(argv[4]), nseq = std::atoi(argv[5]), nstreams = std::atoi(argv[6]);
  const float sigma = std::atof(argv[7]);
  std::ifstream fin(argv[2], std::ios::binary);
  std::vector<uint8_t> gulp((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
  const size_t nbytes = size_t(nchan) * nseq * LWA_SV_INP_PER_CHAN;
  if (gulp.size() != nbytes) {
    std::fprintf(stderr, "gulp has %zu bytes, expected %zu\n", gulp.size(), nbytes);
    return 2;
  }
  ProjTableReader reader(argv[1]);
  auto snap = reader.Poll(3168, nchan, 1ull << 62);
  if (!snap) {
    std::fprintf(stderr, "projector rejected: %s\n", reader.LastError().c_str());
    return 3;
  }
  RfiStage st(0, sigma);
  st.Resize(nchan, nseq);
  if (snap->valid)
    st.SetProjector(snap->ut.data(), snap->vt.data(), snap->q, nchan, snap->sun.empty() ? nullptr : snap->sun.data());
  uint8_t* d = nullptr;
  cudaMalloc(&d, nbytes);
  cudaMemcpy(d, gulp.data(), nbytes, cudaMemcpyHostToDevice);
  std::vector<cudaStream_t> s(nstreams);
  for (auto& x : s) cudaStreamCreate(&x);
  st.NextGulp();
  const int per = nchan / nstreams;
  for (int i = 0; i < nstreams; ++i) {
    st.Run(d, i * per, per, s[i], i);
    st.EndImage(i, s[i]);
  }
  cudaDeviceSynchronize();
  std::vector<uint8_t> out(nbytes);
  cudaMemcpy(out.data(), st.Out(), nbytes, cudaMemcpyDeviceToHost);
  std::ofstream(argv[3], std::ios::binary).write(reinterpret_cast<const char*>(out.data()), nbytes);
  std::printf("q %d sun %d version %llu flagged %llu err %s\n", snap->q, int(!snap->sun.empty()), (unsigned long long)snap->version,
              (unsigned long long)st.CollectFlagged(), cudaGetErrorString(cudaGetLastError()));
  return 0;
}
