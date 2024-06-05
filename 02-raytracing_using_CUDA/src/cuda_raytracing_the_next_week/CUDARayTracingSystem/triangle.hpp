#ifndef TRIANGLE_H
#define TRIANGLE_H
#include "aabb.hpp"
#include "hittable.hpp"
#include "util.hpp"
#include "vec3.hpp"
class Triangle : public Hittable {
 public:
 __device__ Triangle(const Point3 &a, const Vec3 &b, const Vec3 &c,
           Material* mat)
      : plane_origin_{a}, u_{b - a}, v_{c - a}, a_{a}, b_{b}, c_{c}, mat_{mat} {
    set_bounding_box();
    // Finding the Plane That Contains a Given Quadrilateral
    auto n = cross(u_, v_);
    normal_ = unit_vector(n);
    D_ = dot(normal_, plane_origin_);

    area_ = 0.5 * n.length();

    // Orienting Points on The Plane
    w_ = n / dot(n, n);  // a lot of math!!!!
  }

  __device__ void set_bounding_box() {
    auto min_p = mymin(mymin(a_, b_), c_);
    auto max_p = mymax(mymax(a_, b_), c_);
    bbox_ = AABB(min_p, max_p).pad();
  }

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
    Vec3 planar_hitpt_vector = intersection - plane_origin_;
    auto alpha = dot(w_, cross(planar_hitpt_vector, v_));
    auto beta = dot(w_, cross(u_, planar_hitpt_vector));
    rec.p_ = intersection;
    if (!is_interior(alpha, beta, rec)) return false;

    // Ray hits the 2D shape; set the rest of the hit record and return true.
    rec.t_ = t;
    rec.mat_ = mat_;
    rec.set_face_normal(r, normal_);

    return true;
  }

 private:
  __device__ bool is_interior(double a, double b, HitRecord &rec) const {
    // Given the hit point in plane coordinates, return false if it is outside
    // the primitive, otherwise set the hit record UV coordinates and return
    // true.

    if ((a < 0) || (b < 0) || (1 < a + b)) return false;
    float area_a = 0.5 * cross(b_ - rec.p_, c_ - rec.p_).length();
    float area_b = 0.5 * cross(a_ - rec.p_, c_ - rec.p_).length();
    float area_c = 0.5 * cross(a_ - rec.p_, b_ - rec.p_).length();
    float area = area_a + area_b + area_c;
    rec.u_ = area_a / area;
    rec.v_ = area_b / area;
    return true;
  }
  Point3 plane_origin_;
  Vec3 u_, v_;
  Point3 a_, b_, c_;
  Material* mat_;
  AABB bbox_;
  Vec3 normal_;
  double D_;
  Vec3 w_;
  double area_;
};

__device__ Hittable* make_triangle(const Point3 &Q, const Vec3 &u, const Vec3 &v,
                             Material* material) {
  auto ptr = new Triangle{Q, u, v, material};
  memory_recorder_for_hittable->record(ptr);
  return (Hittable*)ptr; 
}
#endif