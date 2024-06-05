#ifndef RANDOM_H
#define RANDOM_H

#include "vec3.hpp"

// https://zhuanlan.zhihu.com/p/376432029
__device__ inline Vec3 random_in_unit_disk(curandState* local_rand_state) {
  float r0 = curand_uniform(local_rand_state);
  float r1 = curand_uniform(local_rand_state);
  return Vec3(sqrt(r0) * cos(2 * PI * r0), sqrt(r1) * sin(2 * PI * r1),
              0.0f);
}

// https://zhuanlan.zhihu.com/p/376432029
__device__ inline Vec3 random_in_unit_sphere(curandState* local_rand_state) {
  float r0 = curand_uniform(local_rand_state);
  float r1 = curand_uniform(local_rand_state);
  float r = curand_uniform(local_rand_state);
  // how to sample points in the volume of sphere with uniform probability
  // https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
  r = powf(r, 0.33333333f);

  return Vec3(2.0f * r * cos(2 * PI * r1) * sqrt(r0 * (1.0f - r0)),
              2.0f * r * sin(2 * PI * r1) * sqrt(r0 * (1.0f - r0)),
              r * (1 - 2.0f * r0));
}

__device__ inline Vec3 random_unit_vector(curandState* local_rand_state) {
  return unit_vector(random_in_unit_sphere(local_rand_state));
}

__device__ inline Vec3 random_on_hemisphere(const Vec3& normal,
                                            curandState* local_rand_state) {
  Vec3 on_unit_sphere = random_unit_vector(local_rand_state);
  if (dot(on_unit_sphere, normal) >
      0.0)  // In the same hemisphere as the normal
    return on_unit_sphere;
  else
    return -on_unit_sphere;
}
#endif