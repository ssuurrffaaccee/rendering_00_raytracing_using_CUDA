#include <stdio.h>

#include "raytracing.h"
#include "hitable_sphere.h"
#include "hitable_list.h"
#include "material_impl.h"
__global__ void initWorldRandomState(curandState* randState) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curand_init(1984, 0, 0, randState);
  }
}
__global__ void initPixelRandomState(int screenWidth, int screenHeight,
                                     curandState* randState) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if ((x >= screenWidth) || (y >= screenHeight)) return;
  int pixelIndex = y * screenWidth + x;
  // Original: Each thread gets same seed, a different sequence number, no
  // offset curand_init(1984, pixel_index, 0, &rand_state[pixel_index]); BUGFIX,
  // Each thread gets different seed, same sequence for performance
  // improvement of about 2x!
  curand_init(1984 + pixelIndex, 0, 0, &randState[pixelIndex]);
}

#define RND (curand_uniform(&localRandState))
__global__ void createWorld(Hitable** dataList, Hitable** world,
                            curandState* randState) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState& localRandState = *randState;
    dataList[0] = (Hitable*)new Sphere(Vec3(0, -1000.0, -1), 1000,
                                       new Lambertian(Vec3(0.5, 0.5, 0.5)));
    int i = 1;
    for (int a = -11; a < 11; a++) {
      for (int b = -11; b < 11; b++) {
        float chooseMat = RND;
        Vec3 center(a + RND, 0.2, b + RND);
        if (chooseMat < 0.8f) {
          dataList[i++] = (Hitable*)new Sphere(
              center, 0.2,
              new Lambertian(Vec3(RND * RND, RND * RND, RND * RND)));
        } else if (chooseMat < 0.95f) {
          dataList[i++] = (Hitable*)new Sphere(
              center, 0.2,
              new Metal(Vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND),
                             0.5f * (1.0f + RND)),
                        0.5f * RND));
        } else {
          dataList[i++] =
              (Hitable*)new Sphere(center, 0.2, new Dielectric(1.5));
        }
      }
    }
    dataList[i++] =
        (Hitable*)new Sphere(Vec3(0, 1, 0), 1.0, new Dielectric(1.5));
    dataList[i++] = (Hitable*)new Sphere(Vec3(-4, 1, 0), 1.0,
                                         new Lambertian(Vec3(0.4, 0.2, 0.1)));
    dataList[i++] = (Hitable*)new Sphere(Vec3(4, 1, 0), 1.0,
                                         new Metal(Vec3(0.7, 0.6, 0.5), 0.0));
    *world = (Hitable*)new HitableList(dataList, 22 * 22 + 1 + 3);
  }
}

__global__ void freeWorld(Hitable** dataList, Hitable** world) {
  for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
    delete ((Sphere*)dataList[i])->matPtr;
    delete dataList[i];
  }
  delete *world;
}

// template <typename R, typename T>
// __device__ R clamp(T value, T min, T max) {
//   if (value < min) {
//     return (R)min;
//   } else if (value > max) {
//     return (R)max;
//   } else {
//     return (R)value;
//   }
// }
// __global__ void raytracingRendering(uchar4* outTexture, int screenWidth,
//                                     int screenHeight, int numberOfSample,
//                                     curandState* randState) {
//   int x = threadIdx.x + blockIdx.x * blockDim.x;
//   int y = threadIdx.y + blockIdx.y * blockDim.y;
//   if ((x >= screenWidth) || (y >= screenHeight)) return;
//   int pixelIndex = y * screenWidth + x;
//   curandState& localRandState = randState[pixelIndex];
//   Vec3 finalColor(curand_uniform(&localRandState),
//                   curand_uniform(&localRandState),
//                   curand_uniform(&localRandState));
//   // randState[pixelIndex] = localRandState;
//   outTexture[pixelIndex] =
//       make_uchar4((unsigned char)(finalColor.x() * 255.0f),
//                   (unsigned char)(finalColor.y() * 255.0f),
//                   (unsigned char)(finalColor.z() * 255.0f), 1);
// }


__device__ Vec3 rayColor(const Ray& r, Hitable** world,
                         curandState* localRandState) {
  Ray curRay = r;
  Vec3 curAttenuation = Vec3(1.0f, 1.0f, 1.0f);
  for (int i = 0; i < 50; i++) {
    HitRecord rec;
    if ((*world)->hit(curRay, 0.001f, FLT_MAX, rec)) {
      Ray scattered;
      Vec3 attenuation;
      if (rec.matPtr->scatter(curRay, rec, attenuation, scattered,
                              localRandState)) {
        curAttenuation *= attenuation;
        curRay = scattered;
      } else {
        return Vec3(0.0, 0.0, 0.0);
      }
    } else {
      // background color
      Vec3 unitDirection = unitVector(curRay.direction());
      float t = 0.5f * (unitDirection.y() + 1.0f);
      Vec3 c = (1.0f - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
      return curAttenuation * c;
    }
  }
  return Vec3(0.0, 0.0, 0.0);  // exceeded recursion
}
__global__ void raytracingRendering(float* accBuffer_, uchar4* outTexture,
                                    int screenWidth, int screenHeight,
                                    CameraRT camera, Hitable** world,
                                    curandState* randState, bool isCameraDirty,
                                    int accTimes) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if ((x >= screenWidth) || (y >= screenHeight)) {
    return;
  }
  int pixelIndex = y * screenWidth + x;
  if (isCameraDirty) {
    accBuffer_[3 * pixelIndex] = 0.0f;
    accBuffer_[3 * pixelIndex + 1] = 0.0f;
    accBuffer_[3 * pixelIndex + 2] = 0.0f;
  }
  curandState& localRandState = randState[pixelIndex];
  Vec3 oneTimeColor(0.0f, 0.0f, 0.0f);
  float u = float(x + curand_uniform(&localRandState)) / float(screenWidth);
  float v = float(y + curand_uniform(&localRandState)) / float(screenHeight);
  Ray r = camera.getRay(u, v, &localRandState);
  oneTimeColor = rayColor(r, world, &localRandState);
  accBuffer_[3 * pixelIndex] += oneTimeColor.x();
  accBuffer_[3 * pixelIndex + 1] += oneTimeColor.y();
  accBuffer_[3 * pixelIndex + 2] += oneTimeColor.z();
  outTexture[pixelIndex] = make_uchar4(
      (unsigned char)(accBuffer_[3 * pixelIndex] / float(accTimes) * 255.0f),
      (unsigned char)(accBuffer_[3 * pixelIndex + 1] / float(accTimes) *
                      255.0f),
      (unsigned char)(accBuffer_[3 * pixelIndex + 2] / float(accTimes) *
                      255.0f),
      1);
}