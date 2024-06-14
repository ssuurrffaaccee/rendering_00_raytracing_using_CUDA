#ifndef INSTANCE_H
#define INSTANCE_H
#include "hittable.hpp"
#include "memory.hpp"
#include "vec3.hpp"
class Translate : public Hittable {
 public:
  __device__ Translate(Hittable* p, const Vec3& displacement)
      : object_(p), offset_(displacement) {
    bbox_ = object_->bounding_box() + offset_;
  }
  __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec,
                      curandState* local_rand_state) const override {
    // Move the ray backwards by the offset
    // Move the ray to the prototype
    Ray offset_r{r.origin() - offset_, r.direction(), r.time()};

    // Determine where (if any) an intersection occurs along the offset ray
    if (!object_->hit(offset_r, ray_t, rec, local_rand_state)) return false;

    // Move the intersection point forwards by the offset
    // Move the intersection point form the prototype to the instance
    rec.p_ += offset_;

    return true;
  }
  __device__ AABB bounding_box() const override { return bbox_; }

 private:
  Hittable* object_;
  Vec3 offset_;
};

__device__ Hittable* make_translate(Hittable* object,
                                    const Vec3& displacement) {
  auto ptr = new Translate{object, displacement};
  memory_recorder_for_hittable->record(ptr);
  return (Hittable*)ptr;
}

class RotateY : public Hittable {
 public:
  __device__ RotateY(Hittable* p, float angle) : object_(p) {
    auto radians = degrees_to_radians(angle);
    sin_theta_ = sin(radians);
    cos_theta_ = cos(radians);
    bbox_ = object_->bounding_box();

    Point3 min{INFINITY_, INFINITY_, INFINITY_};
    Point3 max{-INFINITY_, -INFINITY_, -INFINITY_};
    // iterate on bounding_box vertex, forwrad rotate by angle
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          auto x = i * bbox_.x.max + (1 - i) * bbox_.x.min;
          auto y = j * bbox_.y.max + (1 - j) * bbox_.y.min;
          auto z = k * bbox_.z.max + (1 - k) * bbox_.z.min;
          auto newx = cos_theta_ * x + sin_theta_ * z;
          auto newz = -sin_theta_ * x + cos_theta_ * z;
          Vec3 tester(newx, y, newz);
          for (int c = 0; c < 3; c++) {
            min[c] = mymin(min[c], tester[c]);
            max[c] = mymax(max[c], tester[c]);
          }
        }
      }
    }
    bbox_ = AABB{min, max};
  }
  __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec,
                      curandState* local_rand_state) const override {
    // Change the ray from world space to object space
    auto origin = r.origin();
    auto direction = r.direction();
    // backard rorate by angle to the prototype
    {
      origin[0] = cos_theta_ * r.origin()[0] - sin_theta_ * r.origin()[2];
      origin[2] = sin_theta_ * r.origin()[0] + cos_theta_ * r.origin()[2];

      direction[0] =
          cos_theta_ * r.direction()[0] - sin_theta_ * r.direction()[2];
      direction[2] =
          sin_theta_ * r.direction()[0] + cos_theta_ * r.direction()[2];
    }
    Ray rotated_r{origin, direction, r.time()};

    // Determine where (if any) an intersection occurs in object space
    if (!object_->hit(rotated_r, ray_t, rec, local_rand_state)) return false;

    // forward rorate by angle to the prototype
    {
      // Change the intersection point from object space to world space
      auto p = rec.p_;
      p[0] = cos_theta_ * rec.p_[0] + sin_theta_ * rec.p_[2];
      p[2] = -sin_theta_ * rec.p_[0] + cos_theta_ * rec.p_[2];

      // Change the normal from object space to world space
      auto normal = rec.normal_;
      normal[0] = cos_theta_ * rec.normal_[0] + sin_theta_ * rec.normal_[2];
      normal[2] = -sin_theta_ * rec.normal_[0] + cos_theta_ * rec.normal_[2];
      rec.p_ = p;
      rec.normal_ = normal;
    }
    return true;
  }
  __device__ AABB bounding_box() const override { return bbox_; }

 private:
  Hittable* object_;
  float sin_theta_;
  float cos_theta_;
};

__device__ Hittable* make_rotate_y(Hittable* object, float angle) {
  auto ptr = new RotateY{object, angle};
  memory_recorder_for_hittable->record(ptr);
  return (Hittable*)ptr;
}
#endif