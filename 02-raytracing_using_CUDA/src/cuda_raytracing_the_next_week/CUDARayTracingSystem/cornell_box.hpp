#ifndef CORNEL_BOX_H
#define CORNEL_BOX_H
#include "constant_medium.hpp"
#include "hittable_list.hpp"
#include "instance.hpp"
#include "material.hpp"
#include "quad.hpp"
#include "sphere.hpp"
#include "texture.hpp"
#include "triangle.hpp"
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

__device__ HittableList* pyramid(const Point3 &a, const Point3 &b,
                           Material* mat,
                           Material* triangle_mat) {
  auto sides = new HittableList();
  memory_recorder_for_hittable->record(sides);
  // Construct the two opposite vertices with the minimum and maximum
  // coordinates.
  auto min = Point3{fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z())};
  auto max = Point3{fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z())};

  auto dx = Vec3{max.x() - min.x(), 0.0f, 0.0f};
  auto dy = Vec3{0.0f, max.y() - min.y(), 0.0f};
  auto dz = Vec3{0.0f, 0.0f, max.z() - min.z()};
  auto head = (a + b) * 0.5f + 0.5f * dy;
  auto bottom_0 = Point3{min.x(), min.y(), min.z()};
  auto bottom_1 = bottom_0 + dx;
  auto bottom_2 = bottom_0 + dz + dx;
  auto bottom_3 = bottom_0 + dz;
  sides->add(make_triangle(head, bottom_0, bottom_1, triangle_mat));  // front
  sides->add(make_triangle(head, bottom_1, bottom_2, triangle_mat));  // right
  sides->add(make_triangle(head, bottom_2, bottom_3, triangle_mat));  // back
  sides->add(make_triangle(head, bottom_3, bottom_0, triangle_mat));  // left
  sides->add(
      make_quad(Point3{min.x(), min.y(), min.z()}, dx, dz, mat));  // bottom
  return sides;
}

// HittableList* make_cornell_box() {
//   auto red = make_lambertian_material(Color{.65, .05, .05});
//   auto white = make_lambertian_material(Color{.73, .73, .73});
//   auto green = make_lambertian_material(Color{.12, .45, .15});
//   auto light = make_diffuse_light_material(Color{15, 15, 15});
//   auto world = make_shared<HittableList>();
//   world->add(
//       make_quad(Point3{555, 0, 0}, Vec3{0, 555, 0}, Vec3{0, 0, 555}, green));
//   world->add(make_quad(Point3{0, 0, 0}, Vec3{0, 555, 0}, Vec3{0, 0, 555},
//   red)); world->add(make_quad(Point3{343, 554, 332}, Vec3{-130, 0, 0},
//                        Vec3{0, 0, -105}, light));
//   world->add(
//       make_quad(Point3{0, 0, 0}, Vec3{555, 0, 0}, Vec3{0, 0, 555}, white));
//   world->add(make_quad(Point3{555, 555, 555}, Vec3{-555, 0, 0},
//                        Vec3(0, 0, -555), white));
//   world->add(
//       make_quad(Point3{0, 0, 555}, Vec3{555, 0, 0}, Vec3{0, 555, 0}, white));
//   SPtr<Hittable> box1 = box(Point3{0, 0, 0}, Point3{165, 330, 165}, white);
//   box1 = make_rotate_y(box1, 15);
//   box1 = make_translate(box1, Vec3{265, 0, 295});
//   world->add(box1);

//   SPtr<Hittable> box2 = box(Point3{0, 0, 0}, Point3{165, 165, 165}, white);
//   box2 = make_rotate_y(box2, -18);
//   box2 = make_translate(box2, Vec3{130, 0, 65});
//   world->add(box2);
//   return world;
// }

// HittableList* make_cornell_box_fog() {
//   auto red = make_lambertian_material(Color{.65, .05, .05});
//   auto white = make_lambertian_material(Color{.73, .73, .73});
//   auto green = make_lambertian_material(Color{.12, .45, .15});
//   auto light = make_diffuse_light_material(Color{15, 15, 15});
//   auto world = make_shared<HittableList>();
//   world->add(
//       make_quad(Point3{555, 0, 0}, Vec3{0, 555, 0}, Vec3{0, 0, 555}, green));
//   world->add(make_quad(Point3{0, 0, 0}, Vec3{0, 555, 0}, Vec3{0, 0, 555},
//   red)); world->add(make_quad(Point3{343, 554, 332}, Vec3{-130, 0, 0},
//                        Vec3{0, 0, -105}, light));
//   world->add(
//       make_quad(Point3{0, 0, 0}, Vec3{555, 0, 0}, Vec3{0, 0, 555}, white));
//   world->add(make_quad(Point3{555, 555, 555}, Vec3{-555, 0, 0},
//                        Vec3(0, 0, -555), white));
//   world->add(
//       make_quad(Point3{0, 0, 555}, Vec3{555, 0, 0}, Vec3{0, 555, 0}, white));
//   SPtr<Hittable> box1 = box(Point3{0, 0, 0}, Point3{165, 330, 165}, white);
//   box1 = make_rotate_y(box1, 15);
//   box1 = make_translate(box1, Vec3{265, 0, 295});

//   SPtr<Hittable> box2 = box(Point3{0, 0, 0}, Point3{165, 165, 165}, white);
//   box2 = make_rotate_y(box2, -18);
//   box2 = make_translate(box2, Vec3{130, 0, 65});

//   world->add(make_constant_medium(box1, 0.01, Color{0, 0, 0}));
//   world->add(make_constant_medium(box2, 0.01, Color{1, 1, 1}));
//   return world;
// }
#endif
