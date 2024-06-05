#include <cuda_runtime.h>

#include <cstdio>

#include "helper_cuda.h"

// move x by blockDim.x * gridDim.x, a.k.a move all grid
__global__ void kernel(int *arr, int n) {
  // bounded by n
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    arr[i] = i;
  }
}

int main() {
  int n = 65536;
  int *arr;
  checkCudaErrors(cudaMallocManaged(&arr, n * sizeof(int)));

  kernel<<<32, 128>>>(arr, n);

  checkCudaErrors(cudaDeviceSynchronize());
  for (int i = 0; i < n; i++) {
    printf("arr[%d]: %d\n", i, arr[i]);
  }

  cudaFree(arr);
  return 0;
}
