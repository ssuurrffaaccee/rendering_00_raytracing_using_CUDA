#ifndef TEXTURE_H
#define TEXTURE_H
#include "color.hpp"
// #include "image.hpp"
#include "memory.hpp"
#include "perlin.hpp"
#include "vec3.hpp"
#include "cuda_runtime_api.h"
class Texture {
 public:
  __device__ virtual ~Texture() {};

  __device__ virtual Color value(float u, float v, const Point3 &p) const = 0;
};

__device__ MemoryRecorder<Texture>* memory_recorder_for_texture{nullptr};

class SolidColor : public Texture {
 public:
  __device__ SolidColor(const Color &c) : color_value_(c) {}

  __device__ SolidColor(float red, float green, float blue)
      : SolidColor{Color{red, green, blue}} {}

  __device__ Color value(float u, float v, const Point3 &p) const override {
    return color_value_;
  }

 private:
  Color color_value_;
};
__device__ Texture *make_solid_color_texture(const Color &c) {
  auto ptr = new SolidColor{c};
  memory_recorder_for_texture->record(ptr);
  return (Texture *)ptr;
}
__device__ Texture *make_solid_color_texture(float red, float green,
                                             float blue) {
  auto ptr = new SolidColor{red, green, blue};
  memory_recorder_for_texture->record(ptr);
  return (Texture *)ptr;
}

class CheckerTexture : public Texture {
 public:
  __device__ CheckerTexture(float scale, Texture *even, Texture *odd)
      : inv_scale_{1.0f / scale}, even_{even}, odd_{odd} {}

  __device__ CheckerTexture(float scale, const Color &c1, const Color &c2)
      : inv_scale_{1.0f / scale},
        even_(make_solid_color_texture(c1)),
        odd_(make_solid_color_texture(c2)) {}

  __device__ Color value(float u, float v, const Point3 &p) const override {
    auto x_integer = static_cast<int>(std::floor(inv_scale_ * p.x()));
    auto y_integer = static_cast<int>(std::floor(inv_scale_ * p.y()));
    auto z_integer = static_cast<int>(std::floor(inv_scale_ * p.z()));

    bool is_even = (x_integer + y_integer + z_integer) % 2 == 0;

    return is_even ? even_->value(u, v, p) : odd_->value(u, v, p);
  }

 private:
  float inv_scale_;
  Texture *even_;
  Texture *odd_;
};

__device__ Texture *make_checker_texture(float scale, const Color &c1,
                                         const Color &c2) {
  auto ptr = new CheckerTexture{scale, c1, c2};
  memory_recorder_for_texture->record(ptr);
  return (Texture *)ptr;
}

class UVViewTexture : public Texture {
 public:
  __device__ UVViewTexture(float scale, Texture *even, Texture *odd)
      : scale_{scale}, even_{even}, odd_{odd} {}

  __device__ UVViewTexture(float scale, const Color &c1, const Color &c2)
      : scale_{scale},
        even_(make_solid_color_texture(c1)),
        odd_(make_solid_color_texture(c2)) {}

  __device__ Color value(float u, float v, const Point3 &p) const override {
    auto u_integer = int(scale_ * u);
    auto v_integer = int(scale_ * v);
    bool is_even = (u_integer + v_integer) % 2 == 0;
    return is_even ? even_->value(u, v, p) : odd_->value(u, v, p);
  }

 private:
  float scale_;
  Texture *even_;
  Texture *odd_;
};

__device__ Texture *make_uv_view_texture(float scale, const Color &c1,
                                         const Color &c2) {
  auto ptr = new UVViewTexture{scale, c1, c2};
  memory_recorder_for_texture->record(ptr);
  return (Texture *)ptr;
}
__device__ Color mix(const Color &c0, const Color &c1, float r) {
  Color c_;
  c_.e[0] = (1.0 - r) * c0.e[0] + r * c1.e[0];
  c_.e[1] = (1.0 - r) * c0.e[1] + r * c1.e[1];
  c_.e[2] = (1.0 - r) * c0.e[2] + r * c1.e[2];
  return c_;
}
class ImageTexture : public Texture {
 public:
  __device__ ImageTexture(cudaTextureObject_t texture_object) : texture_object_{texture_object} {
  }
  __device__ Color value(float u, float v, const Point3 &p) const override {
     auto color = tex2D<uchar4>(texture_object_, u, v);
     return Color(color.x/255.0f, color.y/255.0f, color.z/255.0f);
  }
 private:
  cudaTextureObject_t texture_object_;
};

__device__ Texture *make_image_texture(cudaTextureObject_t texture_object) {
  auto ptr = new ImageTexture{texture_object};
  memory_recorder_for_texture->record(ptr);
  return (Texture *)ptr;
}

class NoiseTexture : public Texture {
 public:
  __device__ NoiseTexture(float scale,curandState* local_rand_state) : scale_{scale}, noise{local_rand_state} {}

  __device__ Color value(float u, float v, const Point3 &p) const override {
    auto s = scale_ * p;
    // return Color(1,1,1) * 0.5 * (1 + sin(s.z()));
    // return Color(1,1,1) * 0.5 * (1 + sin(10*noise.turb(s)));
    return Color(1, 1, 1) * 0.5 * (1 + sin(s.z() + 10 * noise.turb(s)));
  }

 private:
  Perlin noise;
  float scale_;
};

__device__ Texture *make_noise_texture(float scale, curandState* local_rand_state) {
  auto ptr = new NoiseTexture{scale, local_rand_state};
  memory_recorder_for_texture->record(ptr);
  return (Texture *)ptr;
}

class TriangleInterpolationTexture : public Texture {
 public:
__device__ TriangleInterpolationTexture(const Color &a_color, const Color &b_color,
                               const Color &c_color)
      : a_color_{a_color}, b_color_{b_color}, c_color_{c_color} {}

__device__ Color value(float u, float v, const Point3 &p) const override {
    return u * a_color_ + v * b_color_ + (1.0f - u - v) * c_color_;
  }

 private:
  Color a_color_, b_color_, c_color_;
};

__device__ Texture* make_triangle_interpolation_texture(const Color &a_color,
                                                  const Color &b_color,
                                                  const Color &c_color) {
  auto ptr = new TriangleInterpolationTexture{a_color, b_color, c_color};
  memory_recorder_for_texture->record(ptr);
  return (Texture *)ptr;
}

#endif