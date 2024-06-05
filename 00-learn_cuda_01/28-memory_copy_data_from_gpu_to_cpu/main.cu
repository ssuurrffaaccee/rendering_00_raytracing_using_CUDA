#include <cuda_runtime.h>

#include <cstdio>

#include "helper_cuda.h"

__global__ void kernel(int *pret) { *pret = 42; }
// cudaMemcpyDeviceToHost
// cudaMemcpyHostToDevice
// cudaMemcpyDeviceToDevice
int main() {
  int *pret;
  checkCudaErrors(cudaMalloc(&pret, sizeof(int)));
  kernel<<<1, 1>>>(pret);
  checkCudaErrors(cudaDeviceSynchronize());

  int ret;
  checkCudaErrors(cudaMemcpy(&ret, pret, sizeof(int), cudaMemcpyDeviceToHost));
  printf("result: %d\n", ret);

  cudaFree(pret);
  return 0;
}
