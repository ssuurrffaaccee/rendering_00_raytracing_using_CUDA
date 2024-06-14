#include <glad/glad.h>

#include "camera.hpp"
#include "raytracing.h"
#include "raytracingDriver.h"
#include "load_texture.hpp"
namespace {

const int BLOCK_SIZE{8};
void createCUDAResource(cudaGraphicsResource_t &cudaResource, GLuint GLtexture,
                        cudaGraphicsMapFlags mapFlags) {
  // Map the GL texture resource with the CUDA resource
  checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaResource, GLtexture,
                                              GL_TEXTURE_2D, mapFlags));
}
void deleteCUDAResource(cudaGraphicsResource_t &cudaResource) {
  if (cudaResource != 0) {
    cudaGraphicsUnregisterResource(cudaResource);
    cudaResource = 0;
  }
}
void allocRanomdState(curandState **pixelRandState,
                      curandState **worldRandState, int screenWidth,
                      int screenHeight) {
  int numPixels = screenWidth * screenHeight;
  checkCudaErrors(
      cudaMalloc((void **)pixelRandState, numPixels * sizeof(curandState)));
  checkCudaErrors(cudaMalloc((void **)worldRandState, 1 * sizeof(curandState)));
}
}  // namespace
RayTracingDriver::RayTracingDriver() {}
RayTracingDriver::~RayTracingDriver() {
  checkCudaErrors(cudaFree(pixelRandState_));
  checkCudaErrors(cudaFree(worldRandState_));
  checkCudaErrors(cudaFree(tranferBuffer_));
  checkCudaErrors(cudaFree(accBuffer_));
  checkCudaErrors(cudaDestroyTextureObject(texture_object_));
  checkCudaErrors(cudaFreeArray(cu_array_));
  freeMemoryRecorder<<<1,1>>>();
  checkCudaErrors(cudaDeviceSynchronize());
  deleteCUDAResource(this->CUDAGraphicsResourceForTexture_);
  cudaDeviceReset();
}
void RayTracingDriver::init(Texture2D &texture, int screenWidth,
                            int screenHeight) {
  {
    // We have to call cudaGLSetGLDevice if we want to use OpenGL
    // interoperability.
    checkCudaErrors(cudaGLSetGLDevice(0));
  }

  {
    createCUDAResource(this->CUDAGraphicsResourceForTexture_, texture.ID,
                       cudaGraphicsMapFlagsWriteDiscard);
    // std::cout<<this->CUDAGraphicsResourceForTexture_<<"\n";
  }
  {
    allocRanomdState(&(this->pixelRandState_), &(this->worldRandState_),
                     screenWidth, screenHeight);
  }
  {
    // we need that 2nd random state to be initialized for the world creation
    initWorldRandomState<<<1, 1>>>(this->worldRandState_);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  }
  {
    int tx = 8;
    int ty = 8;
    // Render our buffer
    dim3 blocks(screenWidth / tx + 1, screenHeight / ty + 1);
    dim3 threads(tx, ty);
    initPixelRandomState<<<blocks, threads>>>(screenWidth, screenHeight,
                                              this->pixelRandState_);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  }
  {
    tranferBufferSize_ = screenWidth * screenHeight * sizeof(uchar4);
    checkCudaErrors(cudaMalloc(&tranferBuffer_, tranferBufferSize_));
    int accBufferSize = screenWidth * screenHeight * sizeof(float) * 3;
    checkCudaErrors(cudaMalloc(&accBuffer_, accBufferSize));
  }
  {
    textureWidth_ = screenWidth;
    textureHeight_ = screenHeight;
  }
  {
    initMemoryRecorder<<<1,1>>>();
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  }
  {
     std::string path = "../../resources/earthmap.jpg";
     auto res = load_cuda_texture_from_file(path,true);
     texture_object_ = res.first;
     cu_array_ = res.second;
  }
  {
    createWorld<<<1, 1>>>(texture_object_,worldRandState_);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  }
}
void RayTracingDriver::render(Camera &camera) {
  // Map the resources so they can be used in the kernel.
  checkCudaErrors(
      cudaGraphicsMapResources(1, &this->CUDAGraphicsResourceForTexture_));
  cudaArray *textureInternalArray{nullptr};
  // Get a device pointer to the OpenGL buffers
  checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(
      &textureInternalArray, this->CUDAGraphicsResourceForTexture_, 0, 0));
  // Compute the grid size
  size_t blocksW = (size_t)ceilf(textureWidth_ / (float)BLOCK_SIZE);
  size_t blocksH = (size_t)ceilf(textureHeight_ / (float)BLOCK_SIZE);
  dim3 gridDim(blocksW, blocksH, 1);
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
  {
    Vec3 lookfrom(camera.position_[0], camera.position_[1],
                  camera.position_[2]);
    glm::vec3 lookat_glm = camera.position_ + camera.front_;
    Vec3 lookat(lookat_glm[0], lookat_glm[1], lookat_glm[2]);
    Vec3 vup(camera.up_[0], camera.up_[1], camera.up_[2]);
    float vfov = 30.0f;
    float aspect = float(textureWidth_) / float(textureHeight_);
    float aperture = 0.0001;
    float focusDist = 10.0f;
    CameraRT cameraRT(lookfrom, lookat, vup, vfov, aspect, aperture, focusDist);
    if (camera.dirty_) {
      renderingAccTimes_ = 1;
    } else {
      renderingAccTimes_ += 1;
    }
    raytracingRendering<<<gridDim, blockDim>>>(
        accBuffer_, tranferBuffer_, this->textureWidth_, this->textureHeight_,
        cameraRT, this->pixelRandState_, camera.dirty_, renderingAccTimes_);
    camera.dirty_ = false;
  }
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  // Copy the destination back to the source array
  checkCudaErrors(cudaMemcpyToArray(textureInternalArray, 0, 0, tranferBuffer_,
                                    tranferBufferSize_,
                                    cudaMemcpyDeviceToDevice));
  // Unmap the resources again so the texture can be rendered in OpenGL
  checkCudaErrors(
      cudaGraphicsUnmapResources(1, &this->CUDAGraphicsResourceForTexture_));
}