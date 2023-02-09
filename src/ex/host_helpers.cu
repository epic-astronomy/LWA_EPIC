#include "host_helpers.h"
#include <cstdint>

/**
 * @brief Pin memory-block to speedup H2D transfers
 * 
 * @tparam T Type of the memory block
 * @param ptr Pointer to the memory block
 * @param nbytes Number of bytes from the beginning to be pinnes
 * @return int 
 */
template<typename T>
int cu_mlock(T* ptr, size_t nbytes){
    cuda_check_err(cudaHostRegister(ptr, nbytes, cudaHostRegisterDefault));
    return 0;
}

template int cu_mlock<uint8_t>(uint8_t*, size_t);