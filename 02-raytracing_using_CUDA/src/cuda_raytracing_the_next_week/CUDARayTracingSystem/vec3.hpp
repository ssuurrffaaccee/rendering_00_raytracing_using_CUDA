#ifndef VEC3_H
#define VEC3_H
#include <cmath>

#include "util.hpp"
using std::fabs;
using std::sqrt;

class Vec3 {
 public:
  float e[3];

  __host__ __device__ Vec3() : e{0, 0, 0} {}
  __host__ __device__ Vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

  __host__ __device__ float x() const { return e[0]; }
  __host__ __device__ float y() const { return e[1]; }
  __host__ __device__ float z() const { return e[2]; }

  __host__ __device__ Vec3 operator-() const {
    return Vec3(-e[0], -e[1], -e[2]);
  }
  __host__ __device__ float operator[](int i) const { return e[i]; }
  __host__ __device__ float &operator[](int i) { return e[i]; }

  __host__ __device__ Vec3 &operator+=(const Vec3 &v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
  }

  __host__ __device__ Vec3 &operator*=(float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
  }

  __host__ __device__ Vec3 &operator/=(float t) { return *this *= 1 / t; }

  __host__ __device__ float length() const { return sqrt(length_squared()); }

  __host__ __device__ float length_squared() const {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
  }

  __host__ __device__ bool near_zero() const {
    // Return true if the vector is close to zero in all dimensions.
    auto s = 1e-8;
    return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
  }

  __device__ static Vec3 random(curandState *local_rand_state) {
    return Vec3(random_float(local_rand_state), random_float(local_rand_state),
                random_float(local_rand_state));
  }

  __device__ static Vec3 random(float min, float max,
                                           curandState *local_rand_state) {
    return Vec3(random_float(min, max, local_rand_state),
                random_float(min, max, local_rand_state),
                random_float(min, max, local_rand_state));
  }
};

// point3 is just an alias for Vec3, but useful for geometric clarity in the
// code.
using Point3 = Vec3;

// Vector Utility Functions

// inline std::ostream& operator<<(std::ostream &out, const Vec3 &v) {
//     return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
// }

__host__ __device__ inline Vec3 operator+(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline Vec3 operator-(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline Vec3 operator*(float t, const Vec3 &v) {
  return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &v, float t) {
  return t * v;
}

__host__ __device__ inline Vec3 operator/(Vec3 v, float t) {
  return (1 / t) * v;
}

__host__ __device__ inline float dot(const Vec3 &u, const Vec3 &v) {
  return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__host__ __device__ inline Vec3 cross(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
              u.e[2] * v.e[0] - u.e[0] * v.e[2],
              u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline Vec3 unit_vector(Vec3 v) { return v / v.length(); }

__device__ inline Vec3 reflect(const Vec3 &v, const Vec3 &n) {
  return v - 2 * dot(v, n) * n;
}

__device__ inline Vec3 refract(const Vec3 &uv, const Vec3 &n,
                                          float etai_over_etat) {
  auto cos_theta = mymin(dot(-uv, n), 1.0);
  Vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
  Vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
  return r_out_perp + r_out_parallel;
}
__device__ inline Vec3 mymin(const Vec3& x, const Vec3& y){
    return Vec3{mymin(x.x(), y.x()),mymin(x.y(), y.y()),mymin(x.z(), y.z())};
}
__device__ inline Vec3 mymax(const Vec3& x, const Vec3& y){
    return Vec3{mymax(x.x(), y.x()),mymax(x.y(), y.y()),mymax(x.z(), y.z())};
}
#endif