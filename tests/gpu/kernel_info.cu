// Prints the launch/resource footprint of the imaging kernel variants
// (block size, shared memory, registers, local-memory spills).
#include <cstdio>
#include "epic_harness.cuh"
using namespace epic_test;
template <typename Acc>
void Report(const char* name, int support) {
  for (int m = kProduction; m <= kGcfRegHalf; ++m) {
    void* k = ImagingKernelFor<FFT128x128, Acc>(support, m);
    cudaFuncAttributes a;
    EPIC_CHECK(cudaFuncGetAttributes(&a, k));
    std::printf("%-18s %-16s support %d: regs %3d, local (spill) %5zu B\n", name, WeightModeName(m),
                support, a.numRegs, a.localSizeBytes);
  }
}
int main() {
  int dev = 0, optin = 0, persm = 0;
  EPIC_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  EPIC_CHECK(cudaDeviceGetAttribute(&persm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
  std::printf("FFT128x128: block_dim (%u, %u, %u) = %u threads, shared_memory_size %u B, launched with %d B dynamic\n",
              FFT128x128::block_dim.x, FFT128x128::block_dim.y, FFT128x128::block_dim.z,
              FFT128x128::block_dim.x * FFT128x128::block_dim.y * FFT128x128::block_dim.z,
              unsigned(FFT128x128::shared_memory_size), int(FFT128x128::shared_memory_size * 1.5));
  std::printf("device: max dynamic smem per block (opt-in) %d B, per SM %d B\n", optin, persm);
  Report<f16_accum_t>("bf16 accumulation", 5);
  Report<float2>("fp32 accumulation", 5);
}
