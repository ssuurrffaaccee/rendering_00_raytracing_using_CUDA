#pragma once
// CUDA headers
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>

#include "cameraSystem/camera.h"
#include "resources/texture.h"
#include "hitable.h"
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
  Hitable** dataList_;
  Hitable** world_;
  int renderingAccTimes_{1};
};