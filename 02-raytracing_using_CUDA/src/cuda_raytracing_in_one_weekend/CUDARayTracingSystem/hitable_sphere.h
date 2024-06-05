#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class Sphere : public Hitable {
 public:
  __device__ Sphere() {}
  __device__ Sphere(Vec3 cen, float r, Material* m)
      : center(cen), radius(r), matPtr(m) {};
  __device__ virtual bool hit(const Ray& r, float tmin, float tmax,
                              HitRecord& rec) const;
  Vec3 center;
  float radius;
  Material* matPtr;
};

__device__ bool Sphere::hit(const Ray& r, float tMin, float tMax,
                            HitRecord& rec) const {
  Vec3 oc = r.origin() - center;
  float a = dot(r.direction(), r.direction());
  float b = dot(oc, r.direction());
  float c = dot(oc, oc) - radius * radius;
  float discriminant = b * b - a * c;
  if (discriminant > 0) {
    float temp = (-b - sqrt(discriminant)) / a;
    if (temp < tMax && temp > tMin) {
      rec.t = temp;
      rec.p = r.at(rec.t);
      rec.normal = (rec.p - center) / radius;
      rec.matPtr = matPtr;
      return true;
    }
    temp = (-b + sqrt(discriminant)) / a;
    if (temp < tMax && temp > tMin) {
      rec.t = temp;
      rec.p = r.at(rec.t);
      rec.normal = (rec.p - center) / radius;
      rec.matPtr = matPtr;
      return true;
    }
  }
  return false;
}

#endif
