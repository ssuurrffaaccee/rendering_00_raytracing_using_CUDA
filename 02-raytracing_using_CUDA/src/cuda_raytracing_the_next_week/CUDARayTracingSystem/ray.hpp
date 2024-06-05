#ifndef RAY_H
#define RAY_H
#include "vec3.hpp"

class Ray {
 public:
  __device__ Ray() {}

  __device__ Ray(const Point3& origin, const Vec3& direction)
      : orig_{origin}, dir_{direction}, tm_{0.0f} {}
  __device__ Ray(const Point3& origin, const Vec3& direction, float offset_time)
      : orig_{origin}, dir_{direction}, tm_{offset_time} {}
  __device__ Point3 origin() const { return orig_; }
  __device__ Vec3 direction() const { return dir_; }
  __device__ float time() const { return tm_; }
  __device__ Point3 at(float t) const { return orig_ + t * dir_; }

 private:
  Point3 orig_;
  Vec3 dir_;
  float tm_;
};
#endif