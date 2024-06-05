#include <cmath>

#include "kernel.hpp"

namespace {
__device__ float fract(float x) {
     return x -int(x); 
}

__device__ float hash(float t) {
  return fract(std::sin(t * 8.233f) * 43758.5453123) > 0.0f ? 0.0f : 1.0f;
}


}  // namespace
__global__ void kernel(uchar4* pixels, int width, int height, int tick) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int pixelIndex = y * width + x;
  float r = hash(float(x/10)) + hash(float(y/10));
  unsigned char gray = (unsigned char)(r/2.0f * 255.0f);
  pixels[pixelIndex] = make_uchar4(gray,gray,gray,255);
}

void kernel_call(uchar4* pixels,int width,int height,int tick){
  dim3 blocks(width / 16, height / 16, 1);
  dim3 threads(16, 16, 1);
  kernel<<<blocks, threads>>>(pixels, width, height, tick);
  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());
}