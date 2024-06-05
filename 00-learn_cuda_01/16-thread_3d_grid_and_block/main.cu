#include <cstdio>
#include <cuda_runtime.h>

// block in gpu as thread in cpu 
// thread in gpu as simd in cpu
__global__ void kernel() {
    printf("Block (%d,%d,%d) of (%d,%d,%d), Thread (%d,%d,%d) of (%d,%d,%d)\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           gridDim.x, gridDim.y, gridDim.z,
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockDim.x, blockDim.y, blockDim.z);
}

int main() {
    kernel<<<dim3(2, 1, 1), dim3(2, 2, 2)>>>();
    cudaDeviceSynchronize();
    return 0;
}