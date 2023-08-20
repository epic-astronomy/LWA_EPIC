/*
 Copyright (c) 2023 The EPIC++ authors

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

#ifndef SRC_EX_HOST_HELPERS_H_
#define SRC_EX_HOST_HELPERS_H_

// #include <glog/logging.h>
#include <cuda_runtime.h>

#include <iostream>

#define cuda_check_err(ans) \
  { gpu_assert((ans), __FILE__, __LINE__); }

/**
 * @brief Error handler for cuda functions
 *
 * @param code cuda result
 * @param file File name
 * @param line Line number
 * @param abort Whether to abort upon failure
 */
inline void gpu_assert(cudaError_t code, const char* file, int line,
                       bool abort = true) {
  if (code != cudaSuccess) {
    // LOG(ERROR) << "GPUassert: " << cudaGetErrorString(code) << file << line;
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

/**
 * @brief Register (pin) the memory allocated on host
 *
 * @tparam T Type of the data
 * @param ptr Pointer to the host data
 * @param nbytes Number of bytes to register
 * @return int Returns 0 on success
 */
template <typename T>
int cu_mlock(T* ptr, size_t nbytes);

/**
 * @brief Unregister host memory
 *
 * @tparam T Type of the data
 * @param ptr Pointer to unregister
 * @return int Returns 0 on success
 */
template <typename T>
int cu_munlock(T* ptr);

/**
 * @brief Get the number of available nVIDia GPUs
 *
 * @return int
 */
int get_ngpus();

#endif  // SRC_EX_HOST_HELPERS_H_
