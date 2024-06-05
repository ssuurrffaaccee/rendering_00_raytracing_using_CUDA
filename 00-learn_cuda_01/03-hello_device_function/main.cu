#include <cstdio>
#include <cuda_runtime.h>

__device__ __inline__ void say_hello() {
    printf("Hello, world!\n");
}

__global__ void kernel() {
    say_hello();
}

int main() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();//stop, then wait gpu to finish
    return 0;
}
