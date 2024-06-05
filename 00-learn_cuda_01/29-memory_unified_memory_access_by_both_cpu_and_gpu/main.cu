#include <cuda_runtime.h>

#include <cstdio>

#include "helper_cuda.h"

__global__ void kernel(int *pret) { *pret = 42; }

int main() {
  int *pret;
  checkCudaErrors(cudaMallocManaged(&pret, sizeof(int)));
  kernel<<<1, 1>>>(pret);  // access by gpu
  checkCudaErrors(cudaDeviceSynchronize());
  printf("result: %d\n",
         *pret);  // access by cpu, auto copied fron gpu by cuda context
  cudaFree(pret);
  return 0;
}
