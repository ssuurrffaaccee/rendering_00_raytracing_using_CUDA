#ifndef UTIL_H
#define UTIL_H
#include <cmath>
#include <cstdlib>
#include <limits>
#include <curand_kernel.h>
using std::sqrt;

// Constants

__device__ const float INFINITY_ = std::numeric_limits<float>::infinity();
__device__ const float PI = 3.1415926535897932385f;

// Utility Functions

__device__ inline float degrees_to_radians(float degrees) {
  return degrees * PI / 180.0;
}

__device__ inline float random_float(curandState* local_rand_state) {
  return curand_uniform(local_rand_state);
}

__device__ inline float random_float(float min, float max,
                                         curandState* local_rand_state) {
  // Returns a random real in [min,max).
  return min + (max - min) * curand_uniform(local_rand_state);
}

__device__ inline int random_int(int min, int max,
                                     curandState* local_rand_state) {
  // Returns a random integer in [min,max].
  return static_cast<int>(random_float(min, max + 0.99999f, local_rand_state));
}

__device__ inline void swap(float& x, float& y){
      float temp = x;
      x = y;
      y = temp;
}

__device__ inline float mymin(float x, float y){
    return x<y?x:y;
}
__device__ inline float mymax(float x, float y){
    return x>y?x:y;
}
#endif