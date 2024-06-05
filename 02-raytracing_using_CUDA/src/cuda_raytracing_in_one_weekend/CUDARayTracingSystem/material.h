#ifndef MATERIAL_H
#define MATERIAL_H
#include "hitable.h"
#include "ray.h"
class Material {
 public:
  __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec,
                                  Vec3& attenuation, Ray& scattered,
                                  curandState* localRandState) const = 0;
};
#endif