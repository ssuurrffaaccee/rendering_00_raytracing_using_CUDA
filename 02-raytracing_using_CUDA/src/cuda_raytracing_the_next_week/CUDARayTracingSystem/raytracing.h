#ifndef RAYTRACING_H
#define RAYTRACING_H
#include <curand_kernel.h>

#include "camera.hpp"

#include "vec3.hpp"
__global__ void initWorldRandomState(curandState *randState);
__global__ void initPixelRandomState(int screenWidth, int screenHeight,
                                     curandState *randState);
__global__ void raytracingRendering(float *acc_buffer, uchar4 *out_texture,
                                    int screen_width, int screen_height,
                                    CameraRT camera,
                                    curandState *randState,
                                    bool is_camera_dirty, int acc_times);
__global__ void initMemoryRecorder();
__global__ void createWorld(cudaTextureObject_t texture_object_,curandState *randState);
__global__ void freeMemoryRecorder();
#endif
