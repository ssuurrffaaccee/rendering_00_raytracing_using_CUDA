#ifndef SPHERE_H
#define SPHERE_H
#include "aabb.hpp"
#include "hittable.hpp"
#include "memory.hpp"
#include "onb.hpp"
#include "util.hpp"
#include "vec3.hpp"
__device__ Vec3 random_to_sphere(float radius, float distance_squared,
                                 curandState* local_rand_state) {
  auto r1 = random_float(local_rand_state);
  auto r2 = random_float(local_rand_state);
  auto z = 1 + r2 * (sqrt(1 - radius * radius / distance_squared) - 1);

  auto phi = 2 * PI * r1;
  auto x = cos(phi) * sqrt(1 - z * z);
  auto y = sin(phi) * sqrt(1 - z * z);

  return Vec3{x, y, z};
}
class Sphere : public Hittable {
 public:
  __device__ Sphere(Point3 center, float radius, Material* mat)
      : center_{center}, radius_{radius}, mat_{mat}, is_moving_{false} {
    auto rvec = Vec3{radius, radius, radius};
    bbox_ = AABB(center_ - rvec, center_ + rvec);
  }
  __device__ Sphere(Point3 center0, Point3 center1, float radius, Material* mat)
      : center_{center0},
        centor_move_vec_{center1 - center0},
        radius_{radius},
        mat_{mat},
        is_moving_{true} {
    auto rvec = Vec3{radius, radius, radius};
    AABB box1(center0 - rvec, center0 + rvec);
    AABB box2(center1 - rvec, center1 + rvec);
    bbox_ = merge(box1, box2);
  }
  __device__ bool hit(const Ray& ray, Interval ray_t, HitRecord& rec,
                      curandState* local_rand_state) const override {
    Point3 centor_on_time = is_moving_ ? sphere_center(ray.time()) : center_;
    Vec3 oc = ray.origin() - centor_on_time;
    auto a = ray.direction().length_squared();
    auto half_b = dot(oc, ray.direction());
    auto c = oc.length_squared() - radius_ * radius_;

    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;

    // Find the nearest root that lies in the acceptable range.
    auto sqrtd = sqrt(discriminant);
    auto root = (-half_b - sqrtd) / a;
    if (!ray_t.surrounds(root)) {
      root = (-half_b + sqrtd) / a;
      if (!ray_t.surrounds(root)) return false;
    }

    rec.t_ = root;
    rec.p_ = ray.at(rec.t_);
    Vec3 outward_normal = (rec.p_ - centor_on_time) / radius_;
    rec.set_face_normal(ray, outward_normal);
    rec.mat_ = mat_;
    float u, v;
    get_sphere_uv(outward_normal, u, v);
    rec.u_ = u;
    rec.v_ = v;
    return true;
  }
  __device__ AABB bounding_box() const override { return bbox_; }
  __device__ float pdf_from(const Point3& origin, const Vec3& sample_direction,
                             curandState* local_rand_state) const override {
    // CHECK(!is_moving_)
    // This method only works for stationary spheres.
    HitRecord rec;
    if (!this->hit(Ray{origin, sample_direction}, Interval{0.001, INFINITY_},
                   rec, local_rand_state)) {
      return 0;
    }

    auto cos_theta_max =
        sqrt(1 - radius_ * radius_ / (center_ - origin).length_squared());
    auto solid_angle = 2 * PI * (1 - cos_theta_max);

    return 1 / solid_angle;
  }
  __device__ Vec3 sample_from(const Vec3& origin,
                              curandState* local_rand_state) const override {
    // CHECK(!is_moving_)
    Vec3 direction = center_ - origin;
    auto distance_squared = direction.length_squared();
    OrthonormalBasis uvw;
    uvw.build_from_w(direction);
    return uvw.local(
        random_to_sphere(radius_, distance_squared, local_rand_state));
  }

 private:
  Point3 center_;
  Point3 centor_move_vec_;
  float radius_;
  Material* mat_;
  bool is_moving_;
  __device__ Point3 sphere_center(float time) const {  // time in [0,1];
    // Linearly interpolate from center1 to center2 according to time, where t=0
    // yields center1, and t=1 yields center2.
    return center_ + time * centor_move_vec_;
  }
  __device__ void get_sphere_uv(const Point3& p, float& u, float& v) const {
    // normal: a given point on the sphere of radius 1.0f, centered at the
    // origin. u: returned value [0,1] of angle around the Y axis from X=-1.
    // horizontal v: returned value [0,1] of angle from Y=-1 to Y=+1. vertical
    //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
    //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
    //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

    auto theta = acos(-p.y());             //[-1,1]->[0,PI]
    auto phi = atan2(-p.z(), p.x()) + PI;  //[R,R]->[-pi,pi];
    u = phi / (2 * PI);
    v = theta / PI;
  }
};

__device__ Hittable* make_static_sphere(const Point3& origin,
                                        const float radius,
                                        Material* material) {
  auto ptr = new Sphere{origin, radius, material};
  memory_recorder_for_hittable->record(ptr);
  return (Hittable*)ptr;
}
__device__ Hittable* make_moving_sphere(const Point3& centor0,
                                        const Point3& centor1,
                                        const float radius,
                                        Material* material) {
  auto ptr = new Sphere{centor0, centor1, radius, material};
  memory_recorder_for_hittable->record(ptr);
  return (Hittable*)ptr;
}
#endif