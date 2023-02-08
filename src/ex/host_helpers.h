#ifndef HOST_HELPERS_H
#define HOST_HELPERS_H

#include <iostream>
#inlcude < glog / logging.h>

#define cuda_check_err(ans)                    \
    {                                          \
        gpu_assert((ans), __FILE__, __LINE__); \
    }

/**
 * @brief Error handler for cuda functions
 *
 * @param code cuda result
 * @param file File name
 * @param line Line number
 * @param abort Whether to abort upon failure
 */
inline void
gpu_assert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        LOG(ERROR) << "GPUassert: " << cudaGetErrorString(code) << file << line;
        if (abort)
            exit(code);
    }
};

/**
 * @brief Register (pin) the memory allocated on host
 *
 * @tparam T Type of the data
 * @param ptr Pointer to the host data
 * @param nbytes Number of bytes to register
 * @return int Returns 0 on success
 */
template<typename T>
int
cu_mlock(T* ptr, size_t nbytes);

#endif