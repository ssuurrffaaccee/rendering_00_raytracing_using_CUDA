#ifndef RAYTRACING_H
#define RAYTRACING_H
#include <curand_kernel.h>

#include "camera.h"
#include "check_cuda.h"
#include "ray.h"
#include "vec3.h"
#include "hitable.h"
__global__ void initWorldRandomState(curandState* randState);
__global__ void initPixelRandomState(int screenWidth, int screenHeight,
                                     curandState* randState);
__global__ void raytracingRendering(float* accBuffer_, uchar4* outTexture,
                                    int screenWidth, int screenHeight,
                                    CameraRT camera, Hitable** world,
                                    curandState* randState, bool isCameraDirty,
                                    int accTimes);
__global__ void createWorld(Hitable** dataList, Hitable** world,
                            curandState* randState);
__global__ void freeWorld(Hitable** dataList, Hitable** world);
#endif
