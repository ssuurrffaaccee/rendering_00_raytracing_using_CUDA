#ifndef HITTABLE_H
#define HITTABLE_H
#include "aabb.hpp"
#include "interval.hpp"
#include "ray.hpp"
#include "vec3.hpp"
#include "memory.hpp"
class Material;
class HitRecord {
 public:
  Point3 p_;
  Vec3 normal_;
  float t_;
  bool front_face_;
  Material* mat_;
  float u_;
  float v_;
  __device__ void set_face_normal(const Ray& ray, const Vec3& outward_normal) {
    // Sets the hit record normal.
    // NOTE: the parameter `outward_normal` is assumed to have unit length.
    front_face_ = dot(ray.direction(), outward_normal) < 0;
    normal_ = front_face_ ? outward_normal : -outward_normal;
  }
};

class Hittable {
 public:
  __device__ virtual ~Hittable() {};

  __device__ virtual bool hit(const Ray& ray, Interval ray_t, HitRecord& rec,
                              curandState* local_rand_state) const = 0;
  __device__ virtual AABB bounding_box() const = 0;
  AABB bbox_;
  bool is_bvh_node_{false};
};
__device__ MemoryRecorder<Hittable>* memory_recorder_for_hittable{nullptr};
#endif