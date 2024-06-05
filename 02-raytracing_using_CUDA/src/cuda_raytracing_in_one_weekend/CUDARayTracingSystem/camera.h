#ifndef CAMEAR_H
#define CAMEAR_H
#include <curand_kernel.h>

#include "ray.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// https://zhuanlan.zhihu.com/p/376432029
inline __device__ Vec3 randomInUnitDisk(curandState* localRandState) {
  float r0 = curand_uniform(localRandState);
  float r1 = curand_uniform(localRandState);
  return Vec3(sqrt(r0) * cos(2 * M_PI * r0), sqrt(r1) * sin(2 * M_PI * r1),
              0.0f);
}
class CameraRT {
 public:
  __host__ __device__
  CameraRT(const Vec3& lookfrom, const Vec3& lookat, const Vec3& vup,
           float vfov, float aspect, float aperture,
           float focusDist) {  // vfov is top to bottom in degrees
    lensRadius = aperture / 2.0f;
    float theta = vfov * ((float)M_PI) / 180.0f;
    float halfHeight = tan(theta / 2.0f);
    float halfWidth = aspect * halfHeight;
    origin = lookfrom;
    w = unitVector(lookfrom - lookat);
    u = unitVector(cross(vup, w));
    v = cross(w, u);
    lowerLeftCorner = origin - halfWidth * focusDist * u -
                      halfHeight * focusDist * v - focusDist * w;
    horizontal = 2.0f * halfWidth * focusDist * u;
    vertical = 2.0f * halfHeight * focusDist * v;
  }
  __device__ Ray getRay(float s, float t,
                                 curandState* local_rand_state) {
    Vec3 rd = lensRadius * randomInUnitDisk(local_rand_state);
    Vec3 offset = u * rd.x() + v * rd.y();
    return Ray(origin + offset, lowerLeftCorner + s * horizontal +
                                    t * vertical - origin - offset);
  }

  Vec3 origin;
  Vec3 lowerLeftCorner;
  Vec3 horizontal;
  Vec3 vertical;
  Vec3 u, v, w;
  float lensRadius;
};
#endif