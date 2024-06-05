#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include "CudaAllocator.h"
#include "ticktock.h"

// visual grid-strip-loop 
// aaaabbbbcccc
//     CUT 
// aaaa bbbb cccc 
//    STACK
//    cccc
//    bbbb
//    aaaa
//   THREAD
//   c c c c
//   b b b b
//   a a a a

// // visual vectorized grid-strip-loop 
// aaaabbbbccccaaaabbbbccccaaaabbbbcccc
//                  CUT 
// aaaa bbbb cccc aaaa bbbb cccc aaaa bbbb cccc
//                VRETICAL
//            a b c a b c a b c
//            a b c a b c a b c
//            a b c a b c a b c
//            a b c a b c a b c
//              Vectorization
//            A B C A B C A B C 
//                  CUT
//            A B C   A B C  A B C 
//                  RENAME  
//            M M M   N N N  P P P
//                  STACK 
//                  P P P
//                  N N N
//                  M M M


__global__ void parallel_sum(int *sum, int const *arr, int n) {
    //Each thread jumps forward blockDim.x * gridDim.x positions from its position each time
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n / 1024; i += blockDim.x * gridDim.x) {
        int local_sum = 0;
        for (int j = i * 1024; j < i * 1024 + 1024; j++) {
            local_sum += arr[j];
        }
        sum[i] = local_sum;
    }
}

int main() {
    int n = 1<<24;
    std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<int, CudaAllocator<int>> sum(n / 1024); // gpu return array

    for (int i = 0; i < n; i++) {
        arr[i] = std::rand() % 4;
    }

    TICK(parallel_sum);
    parallel_sum<<<n / 1024 / 128, 128>>>(sum.data(), arr.data(), n);
    checkCudaErrors(cudaDeviceSynchronize());

    // using cpu to get final result
    int final_sum = 0; 
    for (int i = 0; i < n / 1024; i++) {
        final_sum += sum[i];
    }
    TOCK(parallel_sum);

    printf("result: %d\n", final_sum);

    return 0;
}