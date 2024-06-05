#ifndef QUAD_H
#define QUAD_H
#include "aabb.hpp"
#include "hittable.hpp"
#include "memory.hpp"
#include "util.hpp"
#include "vec3.hpp"
class Quad : public Hittable {
 public:
  __device__ Quad(const Point3& Q, const Vec3& u, const Vec3& v, Material* mat)
      : Q_{Q}, u_{u}, v_{v}, mat_{mat} {
    set_bounding_box();
    // Finding the Plane That Contains a Given Quadrilateral
    auto n = cross(u, v);
    normal_ = unit_vector(n);
    D_ = dot(normal_, Q);

    // Orienting Points on The Plane
    w_ = n / dot(n, n);  // a lot of math!!!!
  }

  __device__ void set_bounding_box() { bbox_ = AABB(Q_, Q_ + u_ + v_).pad(); }

  __device__ AABB bounding_box() const override { return bbox_; }
  __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec,
                      curandState* local_rand_state) const override {
    auto denom = dot(normal_, r.direction());

    // No hit if the ray is parallel to the plane.
    if (fabs(denom) < 1e-8) return false;

    // Return false if the hit point parameter t is outside the ray interval.
    auto t = (D_ - dot(normal_, r.origin())) / denom;
    if (!ray_t.contains(t)) return false;
    // Determine the hit point lies within the planar shape using its plane
    // coordinates.
    auto intersection = r.at(t);
    Vec3 planar_hitpt_vector = intersection - Q_;
    auto alpha = dot(w_, cross(planar_hitpt_vector, v_));
    auto beta = dot(w_, cross(u_, planar_hitpt_vector));

    if (!is_interior(alpha, beta, rec)) return false;

    // Ray hits the 2D shape; set the rest of the hit record and return true.
    rec.t_ = t;
    rec.p_ = intersection;
    rec.mat_ = mat_;
    rec.set_face_normal(r, normal_);

    return true;
  }

 private:
  __device__ bool is_interior(float a, float b, HitRecord& rec) const {
    // Given the hit point in plane coordinates, return false if it is outside
    // the primitive, otherwise set the hit record UV coordinates and return
    // true.

    if ((a < 0) || (1 < a) || (b < 0) || (1 < b)) return false;

    rec.u_ = a;
    rec.v_ = b;
    return true;
  }
  Point3 Q_;
  Vec3 u_, v_;
  Material* mat_;
  Vec3 normal_;
  float D_;
  Vec3 w_;
};

__device__ Hittable* make_quad(const Point3& Q, const Vec3& u, const Vec3& v,
                               Material* material) {
  auto ptr = new Quad{Q, u, v, material};
  memory_recorder_for_hittable->record(ptr);
  return (Hittable*)ptr;
}
#endif