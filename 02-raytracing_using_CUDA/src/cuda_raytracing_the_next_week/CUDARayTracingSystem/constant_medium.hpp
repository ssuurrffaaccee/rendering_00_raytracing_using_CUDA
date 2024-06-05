
#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H
#include "hittable.hpp"
#include "material.hpp"
#include "memory.hpp"
#include "texture.hpp"
class ConstantMedium : public Hittable {
 public:
  __device__ ConstantMedium(Hittable* b, float d, Texture* a)
      : boundary_(b),
        neg_inv_density_(-1 / d),
        phase_function_(make_isotropic_material(a)) {}

  __device__ ConstantMedium(Hittable* b, float d, const Color& c)
      : boundary_(b),
        neg_inv_density_(-1 / d),
        phase_function_(make_isotropic_material(c)) {}

  __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec,
                      curandState* local_rand_state) const override {
    HitRecord rec1, rec2;

    // hit maybe near
    if (!boundary_->hit(r, Interval(-INFINITY_, +INFINITY_), rec1, local_rand_state))
      return false;

    // hit maybe far
    if (!boundary_->hit(r, Interval(rec1.t_ + 0.0001, INFINITY_), rec2,
                        local_rand_state))
      return false;
    // clamp
    if (rec1.t_ < ray_t.min) rec1.t_ = ray_t.min;
    if (rec2.t_ > ray_t.max) rec2.t_ = ray_t.max;
    if (rec1.t_ >= rec2.t_) return false;
    if (rec1.t_ < 0) rec1.t_ = 0;

    auto ray_length = r.direction().length();
    auto distance_inside_boundary = (rec2.t_ - rec1.t_) * ray_length;
    auto hit_distance =
        neg_inv_density_ *
        log(random_float(local_rand_state));  // proceed random distance

    if (hit_distance > distance_inside_boundary) return false;

    rec.t_ = rec1.t_ + hit_distance / ray_length;
    rec.p_ = r.at(rec.t_);

    rec.normal_ = Vec3{1, 0, 0};  // arbitrary
    rec.front_face_ = true;       // also arbitrary
    rec.mat_ = phase_function_;

    return true;
  }

  __device__ AABB bounding_box() const override {
    return boundary_->bounding_box();
  }

 private:
  Hittable* boundary_;
  float neg_inv_density_;
  Material* phase_function_;  // it wrapper other material X, make X filled
                              // in all boundary domain;
};

__device__ Hittable* make_constant_medium(Hittable* b, float d,
                                          const Color& c) {
  auto ptr = new ConstantMedium{b, d, c};
  memory_recorder_for_hittable->record(ptr);
  return (Hittable*)ptr;
}
__device__ Hittable* make_constant_medium(Hittable* b, float d, Texture* a) {
  auto ptr = new ConstantMedium{b, d, a};
  memory_recorder_for_hittable->record(ptr);
  return (Hittable*)ptr;
}
#endif