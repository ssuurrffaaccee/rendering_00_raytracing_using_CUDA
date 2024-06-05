#ifndef RAY_H
#define RAY_H
#include "vec3.h"

class Ray {
 public:
  __device__ Ray() {}
  __device__ Ray(const Vec3& a, const Vec3& b) {
    A = a;
    B = b;
  }
  __device__ Vec3 origin() const { return A; }
  __device__ Vec3 direction() const { return B; }
  __device__ Vec3 at(float t) const { return A + t * B; }

  Vec3 A;
  Vec3 B;
};

#endif