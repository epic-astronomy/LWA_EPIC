#ifndef HOST_HELPERS
#define HOST_HELPERS

// #include <glog/logging.h>
#include <iostream>
#include <cuda_runtime.h>

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
        // LOG(ERROR) << "GPUassert: " << cudaGetErrorString(code) << file << line;
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
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


/**
 * @brief Unregister host memory
 * 
 * @tparam T Type of the data
 * @param ptr Pointer to unregister
 * @return int Returns 0 on success
 */
template<typename T>
int cu_munlock(T* ptr);

/**
 * @brief Get the number of available nVIDia GPUs
 * 
 * @return int 
 */
int get_ngpus();

#endif /* HOST_HELPERS */
