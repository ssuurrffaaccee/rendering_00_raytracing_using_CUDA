#ifndef CORNEL_BOX_H
#define CORNEL_BOX_H
#include "constant_medium.hpp"
#include "hittable_list.hpp"
#include "instance.hpp"
#include "material.hpp"
#include "quad.hpp"
#include "sphere.hpp"
#include "texture.hpp"
// #include "triangle.hpp"
__device__ inline HittableList* box(const Point3& a, const Point3& b,
                                    Material* mat) {
  // Returns the 3D box (six sides) that contains the two opposite vertices a &
  // b.

  auto sides = new HittableList();
  memory_recorder_for_hittable->record(sides);

  // Construct the two opposite vertices with the minimum and maximum
  // coordinates.
  auto min = Point3{mymin(a.x(), b.x()), mymin(a.y(), b.y()), mymin(a.z(), b.z())};
  auto max = Point3{mymax(a.x(), b.x()), mymax(a.y(), b.y()), mymax(a.z(), b.z())};

  auto dx = Vec3{max.x() - min.x(), 0, 0};
  auto dy = Vec3{0, max.y() - min.y(), 0};
  auto dz = Vec3{0, 0, max.z() - min.z()};

  sides->add(
      make_quad(Point3{min.x(), min.y(), max.z()}, dx, dy, mat));  // front
  sides->add(
      make_quad(Point3{max.x(), min.y(), max.z()}, -dz, dy, mat));  // right
  sides->add(
      make_quad(Point3{max.x(), min.y(), min.z()}, -dx, dy, mat));  // back
  sides->add(
      make_quad(Point3{min.x(), min.y(), min.z()}, dz, dy, mat));  // left
  sides->add(
      make_quad(Point3{min.x(), max.y(), max.z()}, dx, -dz, mat));  // top
  sides->add(
      make_quad(Point3{min.x(), min.y(), min.z()}, dx, dz, mat));  // bottom

  return sides;
}

// __device__ HittableList* pyramid(const Point3 &a, const Point3 &b,
//                            Material* mat,
//                            Material* triangle_mat) {
//   auto sides = new HittableList();
//   memory_recorder_for_hittable->record(sides);
//   // Construct the two opposite vertices with the minimum and maximum
//   // coordinates.
//   auto min = Point3{fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z())};
//   auto max = Point3{fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z())};

//   auto dx = Vec3{max.x() - min.x(), 0.0f, 0.0f};
//   auto dy = Vec3{0.0f, max.y() - min.y(), 0.0f};
//   auto dz = Vec3{0.0f, 0.0f, max.z() - min.z()};
//   auto head = (a + b) * 0.5f + 0.5f * dy;
//   auto bottom_0 = Point3{min.x(), min.y(), min.z()};
//   auto bottom_1 = bottom_0 + dx;
//   auto bottom_2 = bottom_0 + dz + dx;
//   auto bottom_3 = bottom_0 + dz;
//   sides->add(make_triangle(head, bottom_0, bottom_1, triangle_mat));  // front
//   sides->add(make_triangle(head, bottom_1, bottom_2, triangle_mat));  // right
//   sides->add(make_triangle(head, bottom_2, bottom_3, triangle_mat));  // back
//   sides->add(make_triangle(head, bottom_3, bottom_0, triangle_mat));  // left
//   sides->add(
//       make_quad(Point3{min.x(), min.y(), min.z()}, dx, dz, mat));  // bottom
//   return sides;
// }

#endif
