#ifndef PERLIN_H
#define PERLIN_H
#include "vec3.hpp"
class Perlin {
 public:
  __device__ Perlin(curandState* local_rand_state) {
    ranvec_ = (Vec3*)malloc(Perlin::point_count * sizeof(Vec3));
    for (int i = 0; i < point_count; ++i) {
      ranvec_[i] = unit_vector(Vec3::random(-1, 1,local_rand_state));
    }
    perm_x_ = perlin_generate_perm(local_rand_state);
    perm_y_ = perlin_generate_perm(local_rand_state);
    perm_z_ = perlin_generate_perm(local_rand_state);
  }
  __device__ ~Perlin() {
    free(ranvec_);
    free(perm_x_);
    free(perm_y_);
    free(perm_z_);
  }
  __device__ float noise(const Point3& p) const {
    // fraction part
    auto u = p.x() - floor(p.x());
    auto v = p.y() - floor(p.y());
    auto w = p.z() - floor(p.z());
    // interger part
    auto i = static_cast<int>(floor(p.x()));
    auto j = static_cast<int>(floor(p.y()));
    auto k = static_cast<int>(floor(p.z()));
    Vec3 c[2][2][2];  // neighbor in 3d space , p in cube center , c is cube's
                      // vertex

    for (int di = 0; di < 2; di++)
      for (int dj = 0; dj < 2; dj++)
        for (int dk = 0; dk < 2; dk++)
          c[di][dj][dk] =
              ranvec_[perm_x_[(i + di) & 255] ^ perm_y_[(j + dj) & 255] ^
                      perm_z_[(k + dk) & 255]];

    return perlin_interp(c, u, v, w);
  }
  // like fourier series sum. weight is decreasing sequence, temp_p is
  // increasing sequence;
  __device__ float turb(const Point3& p, int depth = 7) const {
    auto accum = 0.0;
    auto temp_p = p;
    auto weight = 1.0;
    for (int i = 0; i < depth; i++) {
      accum += weight * noise(temp_p);
      weight *= 0.5;
      temp_p *= 2;
    }
    return fabs(accum);
  }

 private:
  int point_count{256};
  Vec3* ranvec_;
  int* perm_x_;
  int* perm_y_;
  int* perm_z_;

  __device__ int* perlin_generate_perm(curandState* local_rand_state) {
    int* p = (int*)malloc(Perlin::point_count * sizeof(int));
    for (int i = 0; i < Perlin::point_count; i++) {
      p[i] = i;
    }
    permute(p, local_rand_state);
    return p;
  }

  __device__ void permute(int* p,curandState* local_rand_state) {
    for (int i = Perlin::point_count - 1; i > 0; i--) {
      int target = random_int(0, i, local_rand_state);
      int tmp = p[i];
      p[i] = p[target];
      p[target] = tmp;
    }
  }
  // interpolation in cube's internal
  __device__ float perlin_interp(Vec3 c[2][2][2], float u, float v,
                                 float w) const {
    auto uu = u * u * (3 - 2 * u);
    auto vv = v * v * (3 - 2 * v);
    auto ww = w * w * (3 - 2 * w);
    auto accum = 0.0;

    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++) {
          Vec3 weight_v(u - i, v - j, w - k);  // distance vector
          accum += (i * uu + (1 - i) * (1 - uu)) *
                   (j * vv + (1 - j) * (1 - vv)) *
                   (k * ww + (1 - k) * (1 - ww)) * dot(c[i][j][k], weight_v);
        }

    return accum;
  }
};
#endif