#include "host_helpers.h"
#include <cstdint>

template<typename T>
int
cuMLock(T* ptr, size_t nbytes)
{
    cuda_check_err(cudaHostRegister(ptr, nbytes, cudaHostRegisterDefault));
    return 0;
}

template<typename T>
int
cuMUnlock(T* ptr)
{
    cuda_check_err(cudaHostUnregister(ptr));
    return 0;
}

int
get_ngpus()
{
    int n_devices;
    cudaGetDeviceCount(&n_devices);

    return n_devices;
}

template int
cuMLock<uint8_t>(uint8_t*, size_t);
template int
cuMUnlock<uint8_t>(uint8_t*);
template int
cuMLock<float>(float*, size_t);
template int
cuMUnlock<float>(float*);