#ifndef MATERIAL_IMPL_H
#define MATERIAL_IMPL_H
#include "hitable.h"
#include "material.h"
#include "ray.h"
__device__ float schlick(float cosine, float refIdx) {
  float r0 = (1.0f - refIdx) / (1.0f + refIdx);
  r0 = r0 * r0;
  return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool refract(const Vec3& v, const Vec3& n, float niOverNt,
                        Vec3& refracted) {
  Vec3 uv = unitVector(v);
  float dt = dot(uv, n);
  float discriminant = 1.0f - niOverNt * niOverNt * (1 - dt * dt);
  if (discriminant > 0) {
    refracted = niOverNt * (uv - n * dt) - n * sqrt(discriminant);
    return true;
  } else {
    return false;
  }
}
__device__ Vec3 reflect(const Vec3& v, const Vec3& n) {
  return v - 2.0f * dot(v, n) * n;
}

// https://zhuanlan.zhihu.com/p/376432029
__device__ Vec3 randomInUnitSphere(curandState* localRandState) {
  float r0 = curand_uniform(localRandState);
  float r1 = curand_uniform(localRandState);
  float r = curand_uniform(localRandState);
  // how to sample points in the volume of sphere with uniform probability
  // https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
  r = powf(r, 0.33333333f);

  return Vec3(2.0f * r * cos(2 * M_PI * r1) * sqrt(r0 * (1.0f - r0)),
              2.0f * r * sin(2 * M_PI * r1) * sqrt(r0 * (1.0f - r0)),
              r * (1 - 2.0f * r0));
}
class Lambertian : public Material {
 public:
  __device__ Lambertian(const Vec3& a) : albedo(a) {}
  __device__ virtual bool scatter(const Ray& rIn, const HitRecord& rec,
                                  Vec3& attenuation, Ray& scattered,
                                  curandState* localRandState) const {
    Vec3 target = rec.p + rec.normal + randomInUnitSphere(localRandState);
    scattered = Ray(rec.p, target - rec.p);
    attenuation = albedo;
    return true;
  }

  Vec3 albedo;
};

class Metal : public Material {
 public:
  __device__ Metal(const Vec3& a, float f) : albedo(a) {
    if (f < 1)
      fuzz = f;
    else
      fuzz = 1;
  }
  __device__ virtual bool scatter(const Ray& rIn, const HitRecord& rec,
                                  Vec3& attenuation, Ray& scattered,
                                  curandState* localRandState) const {
    Vec3 reflected = reflect(unitVector(rIn.direction()), rec.normal);
    scattered =
        Ray(rec.p, reflected + fuzz * randomInUnitSphere(localRandState));
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0.0f);
  }
  Vec3 albedo;
  float fuzz;
};

class Dielectric : public Material {
 public:
  __device__ Dielectric(float ri) : refIdx(ri) {}
  __device__ virtual bool scatter(const Ray& rIn, const HitRecord& rec,
                                  Vec3& attenuation, Ray& scattered,
                                  curandState* localRandState) const {
    Vec3 outwardNormal;
    Vec3 reflected = reflect(rIn.direction(), rec.normal);
    float niOverNt;
    attenuation = Vec3(1.0, 1.0, 1.0);
    Vec3 refracted;
    float reflectProb;
    float cosine;
    if (dot(rIn.direction(), rec.normal) > 0.0f) {
      outwardNormal = -rec.normal;
      niOverNt = refIdx;
      cosine = dot(rIn.direction(), rec.normal) / rIn.direction().length();
      cosine = sqrt(1.0f - refIdx * refIdx * (1 - cosine * cosine));
    } else {
      outwardNormal = rec.normal;
      niOverNt = 1.0f / refIdx;
      cosine = -dot(rIn.direction(), rec.normal) / rIn.direction().length();
    }
    if (refract(rIn.direction(), outwardNormal, niOverNt, refracted)) {
      reflectProb = schlick(cosine, refIdx);
    } else {
      reflectProb = 1.0f;
    }
    if (curand_uniform(localRandState) < reflectProb) {
      scattered = Ray(rec.p, reflected);
    } else {
      scattered = Ray(rec.p, refracted);
    }
    return true;
  }

  float refIdx;
};
#endif