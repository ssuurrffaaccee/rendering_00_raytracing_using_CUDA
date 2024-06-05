#ifndef CHECK_CUDA_H
#define CHECK_CUDA_H
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

inline void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line) {
  if (result) {

    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) <<" "<<cudaGetErrorString(result)<< " at "
              << file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}
#endif