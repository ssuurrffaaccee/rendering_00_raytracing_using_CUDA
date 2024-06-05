#ifndef AABB_H
#define AABB_H
#include "interval.hpp"
#include "ray.hpp"
#include "vec3.hpp"
class AABB {
 public:
  Interval x, y, z;

  __device__ AABB() {
  }  // The default AABB is empty, since Intervals are empty by default.

  __device__ AABB(const Interval& ix, const Interval& iy, const Interval& iz)
      : x(ix), y(iy), z(iz) {}

  __device__ AABB(const Point3& a, const Point3& b) {
    // Treat the two points a and b as extrema for the bounding box, so we don't
    // require a particular minimum/maximum coordinate order.
    x = Interval(mymin(a[0], b[0]), mymax(a[0], b[0]));
    y = Interval(mymin(a[1], b[1]), mymax(a[1], b[1]));
    z = Interval(mymin(a[2], b[2]), mymax(a[2], b[2]));
  }

  __device__ AABB(const AABB& box0, const AABB& box1) {
    x = merge(box0.x, box1.x);
    y = merge(box0.y, box1.y);
    z = merge(box0.z, box1.z);
  }

  __device__ AABB pad() {
    // Return an AABB that has no side narrower than some delta, padding if
    // necessary.
    float delta = 0.0001;
    Interval new_x = (x.size() >= delta) ? x : x.expand(delta);
    Interval new_y = (y.size() >= delta) ? y : y.expand(delta);
    Interval new_z = (z.size() >= delta) ? z : z.expand(delta);

    return AABB(new_x, new_y, new_z);
  }

  __device__ const Interval& axis(int n) const {
    if (n == 1) return y;
    if (n == 2) return z;
    return x;
  }

  __device__ bool hit(const Ray& r, Interval ray_t) const {
    for (int a = 0; a < 3; a++) {
      auto invD = 1 / r.direction()[a];
      auto orig = r.origin()[a];

      auto t0 = (axis(a).min - orig) * invD;
      auto t1 = (axis(a).max - orig) * invD;

      if (invD < 0) swap(t0, t1);

      if (t0 > ray_t.min) ray_t.min = t0;
      if (t1 < ray_t.max) ray_t.max = t1;

      if (ray_t.max <= ray_t.min) return false;
    }
    return true;
  }
};
__device__ AABB merge(const AABB& box1, const AABB& box2) {
  return AABB{merge(box1.x, box2.x), merge(box1.y, box2.y),
              merge(box1.z, box2.z)};
}
__device__ AABB operator+(const AABB& bbox, const Vec3& offset) {
  return AABB(bbox.x + offset.x(), bbox.y + offset.y(), bbox.z + offset.z());
}

__device__ AABB operator+(const Vec3& offset, const AABB& bbox) {
  return bbox + offset;
}

#endif
