#ifndef MATERIAL_H
#define MATERIAL_H
#include "color.hpp"
#include "hittable.hpp"
#include "memory.hpp"
#include "random.hpp"
#include "ray.hpp"
#include "texture.hpp"
#include "vec3.hpp"
class Material {
 public:
  __device__ virtual ~Material() {};
  __device__ virtual Color emitted(float u, float v, const Point3& p) const {
    return Color{0, 0, 0};
  }
  __device__ virtual bool scatter(const Ray& ray_in, const HitRecord& rec,
                                  Color& attenuation, Ray& scattered,
                                  curandState* local_rand_state) const = 0;
};

__device__ MemoryRecorder<Material>* memory_recorder_for_material{nullptr};

class SimpleDiffuseMaterial : public Material {
 public:
  __device__ bool scatter(const Ray& ray_in, const HitRecord& rec,
                          Color& attenuation, Ray& scattered,
                          curandState* local_rand_state) const override {
    Vec3 direction = random_on_hemisphere(rec.normal_, local_rand_state);
    scattered = Ray{rec.p_, direction, ray_in.time()};
    attenuation = Color{0.5f, 0.5f, 0.5f};
    return true;
  }
};

__device__ Material* make_simple_diffuse_material() {
  auto ptr = new SimpleDiffuseMaterial{};
  memory_recorder_for_material->record(ptr);
  return (Material*)ptr;
}

class Lambertian : public Material {
 public:
  __device__ Lambertian(const Color& a)
      : albedo_{make_solid_color_texture(a)} {}
  __device__ Lambertian(Texture* texture) : albedo_{texture} {}
  __device__ bool scatter(const Ray& r_in, const HitRecord& rec,
                          Color& attenuation, Ray& scattered,
                          curandState* local_rand_state) const override {
    auto scatter_direction =
        rec.normal_ +
        random_unit_vector(
            local_rand_state);  // oppsite may make scatter_direction zero

    // Catch degenerate scatter direction
    if (scatter_direction.near_zero()) {
      scatter_direction = rec.normal_;
    }

    scattered = Ray{rec.p_, scatter_direction, r_in.time()};
    attenuation = albedo_->value(rec.u_, rec.v_, rec.p_);
    return true;
  }

 private:
  Texture* albedo_;
};
__device__ Material* make_lambertian_material(const Color& albedo) {
  auto ptr = new Lambertian{albedo};
  memory_recorder_for_material->record(ptr);
  return (Material*)ptr;
}
__device__ Material* make_lambertian_material(Texture* texture) {
  auto ptr = new Lambertian{texture};
  memory_recorder_for_material->record(ptr);
  return (Material*)ptr;
}
class Metal : public Material {
 public:
  __device__ Metal(const Color& a, float fuzz)
      : fuzz_{fuzz}, albedo_{make_solid_color_texture(a)} {}
  __device__ Metal(Texture* texture, float fuzz) :fuzz_{fuzz}, albedo_{texture} {}

  __device__ bool scatter(const Ray& r_in, const HitRecord& rec,
                          Color& attenuation, Ray& scattered,
                          curandState* local_rand_state) const override {
    Vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal_);
    scattered =
        Ray{rec.p_, reflected + fuzz_ * random_unit_vector(local_rand_state),
            r_in.time()};  // fuzz direction
    attenuation = albedo_->value(rec.u_, rec.v_, rec.p_);
    return (dot(scattered.direction(), rec.normal_) > 0);  // must in same side
    return true;
  }

 private:
  float fuzz_;
  Texture* albedo_;
};
__device__ Material* make_metal_material(const Color& albedo, float fuzzy) {
  auto ptr = new Metal{albedo, fuzzy};
  memory_recorder_for_material->record(ptr);
  return (Material*)ptr;
}

__device__ Material* make_metal_material(Texture* texture, float fuzzy) {
  auto ptr = new Metal{texture, fuzzy};
  memory_recorder_for_material->record(ptr);
  return (Material*)ptr;
}

class Dielectric : public Material {
 public:
  __device__ Dielectric(float index_of_refraction) : ir_{index_of_refraction} {}

  __device__ bool scatter(const Ray& ray_in, const HitRecord& rec,
                          Color& attenuation, Ray& scattered,
                          curandState* local_rand_state) const override {
    attenuation = Color{1.0, 1.0, 1.0};
    float refraction_ratio = rec.front_face_ ? (1.0 / ir_) : ir_;
    Vec3 unit_direction = unit_vector(ray_in.direction());
    Vec3 refracted = refract(unit_direction, rec.normal_, refraction_ratio);
    float cos_theta =
        mymin(dot(-unit_direction, rec.normal_), 1.0);  // for incident angle
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    bool cannot_refract = refraction_ratio * sin_theta > 1.0;  //
    Vec3 direction;
    if (cannot_refract || reflectance(cos_theta, refraction_ratio) >
                              random_float(local_rand_state)) {
      direction = reflect(unit_direction, rec.normal_);
    } else {
      direction = refract(unit_direction, rec.normal_, refraction_ratio);
    }
    scattered = Ray{rec.p_, refracted, ray_in.time()};
    return true;
  }

 private:
  float ir_;  // Index of Refraction
  __device__ static float reflectance(float cosine, float ref_idx) {
    // Use Schlick's approximation for reflectance.
    auto r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
  }
};
__device__ Material* make_dielectric_material(float index_of_refraction) {
  auto ptr = new Dielectric{index_of_refraction};
  memory_recorder_for_material->record(ptr);
  return (Material*)ptr;
}

class DiffuseLight : public Material {
 public:
  __device__ DiffuseLight(Texture* a) : emit(a) {}
  __device__ DiffuseLight(const Color& c) : emit(make_solid_color_texture(c)) {}

  __device__ bool scatter(const Ray& ray_in, const HitRecord& rec,
                          Color& attenuation, Ray& scattered,
                          curandState* local_rand_state) const override {
    return false;
  }

  __device__ Color emitted(float u, float v, const Point3& p) const override {
    return emit->value(u, v, p);
  }

 private:
  Texture* emit;
};

__device__ Material* make_diffuse_light_material(const Color& albedo) {
  auto ptr = new DiffuseLight{albedo};
  memory_recorder_for_material->record(ptr);
  return (Material*)ptr;
}
__device__ Material* make_diffuse_light_material(Texture* texture) {
  auto ptr = new DiffuseLight{texture};
  return (Material*)ptr;
}
// uniform in all direction
class Isotropic : public Material {
 public:
  __device__ Isotropic(const Color& c) : albedo_(make_solid_color_texture(c)) {}
  __device__ Isotropic(Texture* a) : albedo_(a) {}

  __device__ bool scatter(const Ray& ray_in, const HitRecord& rec,
                          Color& attenuation, Ray& scattered,
                          curandState* local_rand_state) const override {
    scattered =
        Ray{rec.p_, random_unit_vector(local_rand_state), ray_in.time()};
    attenuation = albedo_->value(rec.u_, rec.v_, rec.p_);
    return true;
  }

 private:
  Texture* albedo_;
};

__device__ Material* make_isotropic_material(const Color& albedo) {
  auto ptr = new Isotropic{albedo};
  memory_recorder_for_material->record(ptr);
  return (Material*)ptr;
}
__device__ Material* make_isotropic_material(Texture* texture) {
  auto ptr = new Isotropic{texture};
  memory_recorder_for_material->record(ptr);
  return (Material*)ptr;
}
#endif