// Parallel element-wise operations on large float arrays for the pipeline
// kernels (the accumulator adds a 200 MB image 24 times a second with 192
// channels at 256 x 256).
//
// The kernels run on threads pinned to one CPU each (RaftLib), and OpenMP's
// worker threads inherit the affinity of the thread that starts them, so they
// would all share that one CPU. The workers therefore take the process's
// affinity (the main thread's, i.e. the taskset epic++ was started with) the
// first time they run, without the CPUs reserved for the packet receiver
// (cpu_reserve.hpp); the calling thread keeps its own.
#ifndef SRC_EX_PAR_HELPERS_HPP_
#define SRC_EX_PAR_HELPERS_HPP_

#include <omp.h>
#include <sched.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "./cpu_reserve.hpp"

namespace par {

/// CPUs for the workers: the process's, without those reserved for the packet receiver
inline cpu_set_t ProcessCpus() { return cpus::Free(); }

/// In a parallel region: move this worker (not the calling thread) to the process's CPUs, once
inline void FreeWorker() {
  thread_local bool done = false;
  if (!done && omp_get_thread_num() != 0) {
    const cpu_set_t c = ProcessCpus();
    sched_setaffinity(0, sizeof(cpu_set_t), &c);
    done = true;
  }
}

/// dst[i] += src[i], i < n
inline void AddInPlace(float* __restrict dst, const float* __restrict src, size_t n, int nthreads) {
#pragma omp parallel num_threads(nthreads)
  {
    FreeWorker();
#pragma omp for simd schedule(static)
    for (size_t i = 0; i < n; ++i) dst[i] += src[i];
  }
}

/// dst[i] = src[i], i < n
inline void Copy(float* __restrict dst, const float* __restrict src, size_t n, int nthreads) {
#pragma omp parallel num_threads(nthreads)
  {
    FreeWorker();
    const int nt = omp_get_num_threads(), t = omp_get_thread_num();
    const size_t chunk = (n + nt - 1) / nt, a = std::min(n, chunk * t), b = std::min(n, a + chunk);
    if (b > a) std::memcpy(dst + a, src + a, (b - a) * sizeof(float));
  }
}

}  // namespace par

#endif  // SRC_EX_PAR_HELPERS_HPP_
