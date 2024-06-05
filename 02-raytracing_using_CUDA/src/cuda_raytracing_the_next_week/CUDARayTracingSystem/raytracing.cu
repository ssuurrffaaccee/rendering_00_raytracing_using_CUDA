#include <stdio.h>

#include "bvh_node.hpp"
#include "camera.hpp"
#include "cornell_box.hpp"
#include "hittable.hpp"
#include "hittable_list.hpp"
#include "material.hpp"
#include "quad.hpp"
#include "sphere.hpp"
#include "memory.hpp"


__global__ void initMemoryRecorder(){
  if(threadIdx.x == 0 && blockIdx.x == 0){
    memory_recorder_for_hittable = new MemoryRecorder<Hittable>{};
    memory_recorder_for_material = new MemoryRecorder<Material>{};
    memory_recorder_for_texture = new MemoryRecorder<Texture>{};
  }
}

__global__ void freeMemoryRecorder(){
  if(threadIdx.x == 0 && blockIdx.x == 0){
    delete memory_recorder_for_hittable;
    delete memory_recorder_for_material;
    delete memory_recorder_for_texture;
  }
}

__global__ void initWorldRandomState(curandState *randState) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curand_init(1984, 0, 0, randState);
  }
}
__global__ void initPixelRandomState(int screenWidth, int screenHeight,
                                     curandState *randState) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if ((x >= screenWidth) || (y >= screenHeight)) return;
  int pixelIndex = y * screenWidth + x;
  // Original: Each thread gets same seed, a different sequence number, no
  // offset curand_init(1984, pixel_index, 0, &rand_state[pixel_index]); BUGFIX,
  // Each thread gets different seed, same sequence for performance
  // improvement of about 2x!
  curand_init(1984 + pixelIndex, 0, 0, &randState[pixelIndex]);
}
///////////////////////////////////////////
__device__ HittableList *G_WORLD{nullptr};
///////////////////////////////////////////

#define RND (curand_uniform(&local_rand_state))

__device__ void final_scene(cudaTextureObject_t texture_object, curandState *local_rand_state) {

  G_WORLD = new HittableList{};
  memory_recorder_for_hittable->record(G_WORLD);
  {
    HittableList *boxes1 = new HittableList{};
    memory_recorder_for_hittable->record(boxes1);
    { // ground
      auto ground = make_lambertian_material(Color{0.48f, 0.83f, 0.53f});
      int boxes_per_side = 20;
      for (int i = 0; i < boxes_per_side; i++)
      {
        for (int j = 0; j < boxes_per_side; j++)
        {
          auto w = 100.0f;
          auto x0 = -1000.0f + i * w;
          auto z0 = -1000.0f + j * w;
          auto y0 = 0.0f;
          auto x1 = x0 + w;
          auto y1 = random_float(1, 101, local_rand_state);
          auto z1 = z0 + w;
          boxes1->add(box(Point3{x0, y0, z0}, Point3{x1, y1, z1}, ground));
        }
      }
    }
    G_WORLD->add(make_bvh_node(boxes1, local_rand_state));
  }
  // {
  //   auto ground = make_lambertian_material(Color{0.48f, 0.83f, 0.53f});
  //   G_WORLD->add(make_quad(Point3{-1000.0f, 0.0f, -1000.0f}, Vec3{2000, 0, 0},
  //                          Vec3{0, 0, 2000}, ground));
  // }
  {
    auto background = make_lambertian_material(Color{0.83f,0.48f,0.53f});
    G_WORLD->add(make_quad(Point3{-1000.0f, 0.0f, 1000.0f}, Vec3{2000.0f, 0, 0},
                           Vec3{0, 2000.0f, 0.0f}, background));  
  }
  {  // light
    auto light = make_diffuse_light_material(Color{7, 7, 7});
    G_WORLD->add(make_quad(Point3{123, 554, 147}, Vec3{300, 0, 0},
                           Vec3{0, 0, 265}, light));
  }
  {  // motion blur
    auto center1 = Point3{400, 400, 200};
    auto center2 = center1 + Vec3{30, 0, 0};
    auto sphere_material = make_lambertian_material(Color{0.7f, 0.3f, 0.1f});
    G_WORLD->add(make_moving_sphere(center1, center2, 50, sphere_material));
  }
  // // dielectric
  G_WORLD->add(make_static_sphere(Point3{260, 150, 45}, 50,
                                  make_dielectric_material(1.5)));
  // metal
  {
    G_WORLD->add(make_static_sphere(
        Point3{0, 150, 145}, 50, make_metal_material(Color{0.8f, 0.8f, 0.9f}, 1.0f)));
  }

  // volume
  {
    auto boundary = make_static_sphere(Point3{360, 150, 145}, 70,
                                       make_dielectric_material(1.5));
    G_WORLD->add(boundary);
    G_WORLD->add(make_constant_medium(boundary, 0.2f, Color{0.2f, 0.4f, 0.9f}));
    // boundary = make_static_sphere(Point3{0, 0, 0}, 5000,
    //                               make_dielectric_material(1.5));
    // G_WORLD->add(make_constant_medium(boundary, .0001f, Color{1, 1, 1}));
  }
  {
  // texture
    auto emat = make_lambertian_material(
        make_image_texture(texture_object));
    G_WORLD->add(make_static_sphere(Point3{400, 200, 400}, 100, emat));
  }
  {
    // perlin noise
    auto pertext = make_noise_texture(0.1f, local_rand_state);
    G_WORLD->add(make_static_sphere(Point3{220, 280, 300}, 80,
                                  make_lambertian_material(pertext)));
  }
 //  
  // // instance
  {
    HittableList *boxes2 = new HittableList{};
    memory_recorder_for_hittable->record(boxes2);
    auto white = make_lambertian_material(Color{.73f, .73f, .73f});
    int ns = 1000;
    for (int j = 0; j < ns; j++){
      boxes2->add(make_static_sphere(Point3::random(0, 165, local_rand_state), 10, white));
    }
    G_WORLD->add(make_translate(
        make_rotate_y(make_bvh_node(boxes2, local_rand_state), 15),
        Vec3{-100, 270, 395}));
  }

  // {
  //   auto white = make_lambertian_material(Color{.73f, .73f, .73f});
  //   // auto white_another = make_metal_material(Color{0.8f, 0.8f, 0.9f}, 0.5f);
  //   // auto white = make_dielectric_material(1.5);
  //   auto box_ptr = box(Point3{-100.0f, 270.0f, 395.0f}, Point3{65.0f, 435.0f, 565.0f}, white);
  //   auto rotated_box_ptr = make_rotate_y(box_ptr, 15.0f);
  //   G_WORLD->add(make_constant_medium(rotated_box_ptr, 0.01, Color{1, 1, 1}));
  //   // auto box_sky_ptr = box(Point3{-1000.0f, 0.0f, -1000.0f}, Point3{1000.0f, 600.0f, 1000.0f}, white);
  //   // G_WORLD->add(make_rotate_y(box_sky_ptr, 15.0f));
  // }
  // {
  //     auto triangle_texture = make_triangle_interpolation_texture(
  //     Color{1.0f, 0.0f, 0.0f}, Color{0.0f, 1.0f, 0.0f},
  //     Color{0.0f, 0.0f, 1.0f});
  //     auto triangle_material = make_lambertian_material(triangle_texture);
  //     // auto triangle_material =make_metal_material(triangle_texture, 0.8f);
  //     Hittable* box2 = pyramid(Point3{300.0f, 100.0f, 300.0f}, Point3{500.0f, 300.0f, 500.0f}, triangle_material, triangle_material);
  //     G_WORLD->add(make_rotate_y(box2, 15.0f));
  // }
}
__global__ void createWorld(cudaTextureObject_t texture_object,curandState *randState){
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    final_scene(texture_object, randState);
  }
}

__device__ Color ray_color(const Ray &ray, const int max_depth,
                           Hittable* world, const Color &background_color,
                           curandState *local_rand_state) {
  Ray cur_ray = ray;
  Color final_color{0.0f, 0.0f, 0.0f};
  Color acc_attenuation{1.0f, 1.0f, 1.0f};
  int depth = 0;
  for (; depth < max_depth; depth++) {
    HitRecord rec;
    if (world->hit(cur_ray, Interval{0.001, INFINITY_}, rec, local_rand_state)) {
      Ray scattered;
      Color attenuation;
      Color color_from_emission = rec.mat_->emitted(rec.u_, rec.v_, rec.p_);
      if (rec.mat_->scatter(cur_ray, rec, attenuation,scattered,local_rand_state)) {  // ray path continue
        final_color += acc_attenuation * color_from_emission;
        acc_attenuation = acc_attenuation * attenuation;
        cur_ray = scattered;
        continue;
      } else {  // ray path finish
        final_color += acc_attenuation * color_from_emission;
        break;
      }
    } else {  // ray path finish
      final_color +=
          acc_attenuation * background_color;  // light from background
      break;
    }
  }
  if (depth >= max_depth) {
    return Color{0, 0, 0};
  }
  return final_color;
}

__device__ float clamp(float x){
  if (x < 0.00f) return 0.00f;
  if (x > 0.999f) return 0.999;
  return x;
}
const int BROKEN_SIZE_IN_PIXELS = 10;
const int BROKEN_SIZE_IN_PIXELS_POW_2 = BROKEN_SIZE_IN_PIXELS*BROKEN_SIZE_IN_PIXELS;
__global__ void raytracingRendering(float *acc_buffer, uchar4 *out_texture,
                                    int screen_width, int screen_height,
                                    CameraRT camera, curandState *randState,
                                    bool is_camera_dirty, int acc_times) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if ((x >= screen_width) || (y >= screen_height)) {
    return;
  }
  int pixelIndex = y * screen_width + x;
  if (is_camera_dirty) {
    acc_buffer[3 * pixelIndex] = 0.0f;
    acc_buffer[3 * pixelIndex + 1] = 0.0f;
    acc_buffer[3 * pixelIndex + 2] = 0.0f;
  }
  curandState &local_rand_state = randState[pixelIndex];
  Vec3 one_time_color(0.0f, 0.0f, 0.0f);
  int index_in_pixel = acc_times % BROKEN_SIZE_IN_PIXELS_POW_2;
  // float u = float(x + index_in_pixel*curand_uniform(&local_rand_state)/BROKEN_SIZE_IN_PIXELS) / float(screen_width);
  // float v = float(y + index_in_pixel*curand_uniform(&local_rand_state)/BROKEN_SIZE_IN_PIXELS) / float(screen_height);
  float u = float(x + (index_in_pixel/BROKEN_SIZE_IN_PIXELS+ curand_uniform(&local_rand_state))/float(BROKEN_SIZE_IN_PIXELS)) / float(screen_width);
  float v = float(y + (index_in_pixel%BROKEN_SIZE_IN_PIXELS+ curand_uniform(&local_rand_state))/float(BROKEN_SIZE_IN_PIXELS)) / float(screen_height);
  Ray r = camera.get_ray(u, v, &local_rand_state);
  one_time_color =
      ray_color(r, 50, G_WORLD, Color{0.0, 0.0, 0.0}, &local_rand_state);
  acc_buffer[3 * pixelIndex] += one_time_color.x();
  acc_buffer[3 * pixelIndex + 1] += one_time_color.y();
  acc_buffer[3 * pixelIndex + 2] += one_time_color.z();
  out_texture[pixelIndex] = make_uchar4(
      (unsigned char)(clamp(sqrt(acc_buffer[3 * pixelIndex] / float(acc_times))) * 255.0f),
      (unsigned char)(clamp(sqrt(acc_buffer[3 * pixelIndex + 1] / float(acc_times))) *
                      255.0f),
      (unsigned char)(clamp(sqrt(acc_buffer[3 * pixelIndex + 2] / float(acc_times))) *
                      255.0f),
      1);
}