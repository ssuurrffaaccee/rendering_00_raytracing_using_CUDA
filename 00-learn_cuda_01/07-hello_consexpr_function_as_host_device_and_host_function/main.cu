#include <cstdio>
#include <cuda_runtime.h>

// need --expt-relaxed-constexpr
constexpr const char *cuthead(const char *p) {
    return p + 1;
}

__global__ void kernel() {
    printf(cuthead("Gello, world!\n"));
}

int main() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf(cuthead("Cello, world!\n"));
    return 0;
}
