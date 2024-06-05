#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "helper_cuda.h"

template <class T>
struct CudaAllocator {
  using value_type = T;
  
  CudaAllocator() = default;
  template<class _Other>
  constexpr CudaAllocator(const CudaAllocator<_Other>&) noexcept {}
  
  T *allocate(size_t size) {
    T *ptr = nullptr;
    checkCudaErrors(cudaMallocManaged(&ptr, size * sizeof(T)));
    return ptr;
  }

  void deallocate(T *ptr, size_t size = 0) { checkCudaErrors(cudaFree(ptr)); }

  template <class... Args>
  void construct(T *p, Args &&...args) {
    if constexpr (!(sizeof...(Args) == 0 && std::is_pod_v<T>)) {
      // init when no arg or is_pod
      // unified memory so cpu can access
      // copy data from gpu to cpu, it's slow
      ::new ((void *)p) T(std::forward<Args>(args)...);
    }
  }
};

__global__ void kernel(int *arr, int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    arr[i] = i;
  }
}

int main() {
  int n = 65536;
  std::vector<int, CudaAllocator<int>> arr(n);

  kernel<<<32, 128>>>(arr.data(), n);

  checkCudaErrors(cudaDeviceSynchronize());
  for (int i = 0; i < n; i++) {
    printf("arr[%d]: %d\n", i, arr[i]);
  }

  return 0;
}