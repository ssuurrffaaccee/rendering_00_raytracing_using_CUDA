#include <stdio.h>

#include "bvh_node.hpp"
#include "camera.hpp"
#include "cornell_box.hpp"
#include "hittable.hpp"
#include "hittable_list.hpp"
#include "material.hpp"
#include "memory.hpp"
#include "pdf.hpp"
#include "quad.hpp"
#include "sphere.hpp"

__global__ void initMemoryRecorder() {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    memory_recorder_for_hittable = new MemoryRecorder<Hittable>{};
    memory_recorder_for_material = new MemoryRecorder<Material>{};
    memory_recorder_for_texture = new MemoryRecorder<Texture>{};
  }
}

__global__ void freeMemoryRecorder() {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
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
__device__ HittableList *G_LIGHT{nullptr};
///////////////////////////////////////////

#define RND (curand_uniform(&local_rand_state))
__device__ void make_cornell_box_with_explicit_lights_plus_gloss_shpere() {
  G_WORLD = new HittableList{};
  memory_recorder_for_hittable->record(G_WORLD);
  G_LIGHT = new HittableList{};
  memory_recorder_for_hittable->record(G_LIGHT);
  auto red = make_lambertian_material(Color{.65, .05, .05});
  auto white = make_lambertian_material(Color{.73, .73, .73});
  auto green = make_lambertian_material(Color{.12, .45, .15});
  auto light = make_diffuse_light_material(Color{15, 15, 15});
  ////////////////just for sampling
  auto empty_material = new Material();
  memory_recorder_for_material->record(empty_material);
  G_LIGHT->add(
      make_quad(Point3{343, 554, 332}, Vec3{-130, 0, 0}, Vec3{0, 0, -105}, empty_material));
  G_LIGHT->add(make_static_sphere(Point3{190, 90, 190}, 90, empty_material));
  ///////////////
  G_WORLD->add(
      make_quad(Point3{555, 0, 0}, Vec3{0, 555, 0}, Vec3{0, 0, 555}, green));
  G_WORLD->add(make_quad(Point3{0, 0, 0}, Vec3{0, 555, 0}, Vec3{0, 0, 555}, red));
  G_WORLD->add(make_quad(Point3{343, 554, 332}, Vec3{-130, 0, 0},
                       Vec3{0, 0, -105}, light));
  G_WORLD->add(
      make_quad(Point3{0, 0, 0}, Vec3{555, 0, 0}, Vec3{0, 0, 555}, white));
  G_WORLD->add(make_quad(Point3{555, 555, 555}, Vec3{-555, 0, 0},
                       Vec3(0, 0, -555), white));
  G_WORLD->add(
      make_quad(Point3{0, 0, 555}, Vec3{555, 0, 0}, Vec3{0, 555, 0}, white));
  Hittable* box1 = box(Point3{0, 0, 0}, Point3{165, 330, 165}, white);
  box1 = make_rotate_y(box1, 15);
  box1 = make_translate(box1, Vec3{265, 0, 295});
  G_WORLD->add(box1);
  // Glass Sphere
  auto glass = make_dielectric_material(1.5);
  G_WORLD->add(make_static_sphere(Point3{190, 90, 190}, 90, glass));
}
__device__ void final_scene(cudaTextureObject_t texture_object,
                            curandState *local_rand_state) {
  make_cornell_box_with_explicit_lights_plus_gloss_shpere();
}
__global__ void createWorld(cudaTextureObject_t texture_object,
                            curandState *randState) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    final_scene(texture_object, randState);
  }
}

__device__ Color ray_color(const Ray &ray, const int max_depth, Hittable *world,
                           Hittable *lights, const Color &background_color,
                           curandState *local_rand_state) {
  Ray cur_ray = ray;
  Color final_color{0.0f, 0.0f, 0.0f};
  // Color acc_attenuation{1.0f, 1.0f, 1.0f};
  Color acc_attenuation_times_scattering_over_sampling_pdf{1.0f, 1.0f, 1.0f};
  int depth = 0;
  HittablePDF shared_light_hittable_pdf{};
  for (; depth < max_depth; depth++) {
    HitRecord rec;
    if (world->hit(cur_ray, Interval{0.001, INFINITY_}, rec, local_rand_state)) {
      // Ray scattered;
      // Color attenuation;
      Color color_from_emission =
          rec.mat_->emitted(cur_ray, rec, rec.u_, rec.v_, rec.p_);
      ScatterRecord scatter_record;
      if (rec.mat_->scatter(cur_ray, rec,
                            scatter_record, local_rand_state)) {  // ray path continue
        if (scatter_record.skip_pdf_) {
          acc_attenuation_times_scattering_over_sampling_pdf =
              acc_attenuation_times_scattering_over_sampling_pdf *
              scatter_record.attenuation_;
          cur_ray = scatter_record.skip_pdf_ray_;
          // no final_color += ..., just perfect specular, aka scattering_pdf=1
          // and nex_ray_sampling_pdf=1
          // and color_from_emission = Color{0, 0, 0};
          continue;
        }
        shared_light_hittable_pdf.object_ = lights;
        shared_light_hittable_pdf.origin_ = rec.p_;
        PDF *next_ray_samplling_pdf_ptr =
            scatter_record.get_next_ray_samplling_pdf_ptr();
        MixturePDF mixed_pdf{&shared_light_hittable_pdf,
                             next_ray_samplling_pdf_ptr};
        Vec3 scatter_direction = mixed_pdf.sample(local_rand_state);
        // printf("%f,%f,%f\n",scatter_direction.x(),scatter_direction.y(),scatter_direction.z());
        Ray scattered = Ray{rec.p_, scatter_direction, ray.time()};
        float nex_ray_sampling_pdf =
            mixed_pdf.pdf(scattered.direction(), local_rand_state);
        auto scattering_pdf = rec.mat_->scattering_pdf(ray, rec, scattered);
        final_color += acc_attenuation_times_scattering_over_sampling_pdf *
                       color_from_emission;
        acc_attenuation_times_scattering_over_sampling_pdf =
            acc_attenuation_times_scattering_over_sampling_pdf *
            (scatter_record.attenuation_ * scattering_pdf /
             nex_ray_sampling_pdf);
        cur_ray = scattered;
        continue;
      } else {  // ray path finish
        final_color += acc_attenuation_times_scattering_over_sampling_pdf *
                       color_from_emission;
        break;
      }
    } else {  // ray path finish
      final_color += acc_attenuation_times_scattering_over_sampling_pdf *
                     background_color;  // light from background
      break;
    }
  }
  if (depth >= max_depth) {
    return Color{0, 0, 0};
  }
  return final_color;
}

__device__ float clamp(float x) {
  if (x < 0.00f) return 0.00f;
  if (x > 0.999f) return 0.999;
  return x;
}
const int BROKEN_SIZE_IN_PIXELS = 10;
const int BROKEN_SIZE_IN_PIXELS_POW_2 =
    BROKEN_SIZE_IN_PIXELS * BROKEN_SIZE_IN_PIXELS;
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
  // float u = float(x +
  // index_in_pixel*curand_uniform(&local_rand_state)/BROKEN_SIZE_IN_PIXELS) /
  // float(screen_width); float v = float(y +
  // index_in_pixel*curand_uniform(&local_rand_state)/BROKEN_SIZE_IN_PIXELS) /
  // float(screen_height);
  float u = float(x + (index_in_pixel / BROKEN_SIZE_IN_PIXELS +
                       curand_uniform(&local_rand_state)) /
                          float(BROKEN_SIZE_IN_PIXELS)) /
            float(screen_width);
  float v = float(y + (index_in_pixel % BROKEN_SIZE_IN_PIXELS +
                       curand_uniform(&local_rand_state)) /
                          float(BROKEN_SIZE_IN_PIXELS)) /
            float(screen_height);
  Ray r = camera.get_ray(u, v, &local_rand_state);
  one_time_color =
      ray_color(r, 50, G_WORLD, G_LIGHT, Color{0.0, 0.0, 0.0}, &local_rand_state);
  acc_buffer[3 * pixelIndex] += one_time_color.x();
  acc_buffer[3 * pixelIndex + 1] += one_time_color.y();
  acc_buffer[3 * pixelIndex + 2] += one_time_color.z();
  out_texture[pixelIndex] =
      make_uchar4((unsigned char)(clamp(sqrt(acc_buffer[3 * pixelIndex] /
                                             float(acc_times))) *
                                  255.0f),
                  (unsigned char)(clamp(sqrt(acc_buffer[3 * pixelIndex + 1] /
                                             float(acc_times))) *
                                  255.0f),
                  (unsigned char)(clamp(sqrt(acc_buffer[3 * pixelIndex + 2] /
                                             float(acc_times))) *
                                  255.0f),
                  1);
}