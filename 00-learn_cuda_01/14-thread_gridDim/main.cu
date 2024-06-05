#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel() {
    printf("Block %d of %d, Thread %d of %d\n",
           blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);
}
// Idx       in  Dim 
// threadIdx in  blockDim 
// blockIdx  in  gridDim

int main() {
    kernel<<<2, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
