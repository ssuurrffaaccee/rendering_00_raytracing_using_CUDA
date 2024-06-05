#ifndef HITABLE_H
#define HITABLE_H
#include "ray.h"

class Material;

struct HitRecord {
  float t;
  Vec3 p;
  Vec3 normal;
  Material* matPtr;
};

struct Hitable {
 public:
  __device__ virtual bool hit(const Ray& r, float tMin, float tMax,
                              HitRecord& rec) const = 0;
};
#endif