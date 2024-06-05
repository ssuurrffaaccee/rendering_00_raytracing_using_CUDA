#pragma once
// CUDA headers
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>
#include "check_cuda.h"
#include "cameraSystem/camera.h"
#include "resources/texture.h"
#include "resources/load_texture.hpp"
struct RayTracingDriver {
  RayTracingDriver();
  ~RayTracingDriver();
  void init(Texture2D& texture, int screenWidth, int screenHeight);
  void render(Camera& camera);

 private:
  cudaGraphicsResource_t CUDAGraphicsResourceForTexture_{0};
  curandState* pixelRandState_;
  curandState* worldRandState_;
  int textureWidth_{0};
  int textureHeight_{0};
  uchar4* tranferBuffer_{nullptr};
  int tranferBufferSize_{0};
  float* accBuffer_{nullptr};
  int renderingAccTimes_{1};
  cudaTextureObject_t texture_object_;
  cudaArray *cu_array_;
};