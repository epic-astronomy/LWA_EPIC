#include "host_helpers.h"
#include <cstdint>

template<typename T>
int cu_mlock(T* ptr, size_t nbytes){
    cuda_check_err(cudaHostRegister(ptr, nbytes, cudaHostRegisterDefault));
    return 0;
}

template int cu_mlock<uint8_t>(uint8_t*, size_t);