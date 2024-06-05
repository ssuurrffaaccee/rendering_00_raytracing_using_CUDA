#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel() {
    printf("Hello, world!\n");
}

int main() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();//stop, then wait gpu to finish
    return 0;
}
