#ifndef CAMEAR_H
#define CAMEAR_H
#include "random.hpp"
#include "ray.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class CameraRT {
 public:
  __host__ __device__
  CameraRT(const Vec3& lookfrom, const Vec3& lookat, const Vec3& vup,
           float vfov, float aspect, float aperture,
           float focus_dist) {  // vfov is top to bottom in degrees
    lens_radius = aperture / 2.0f;
    float theta = vfov * ((float)M_PI) / 180.0f;
    float half_height = tan(theta / 2.0f);
    float half_width = aspect * half_height;
    origin = lookfrom;
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);
    lower_left_corner = origin - half_width * focus_dist * u -
                        half_height * focus_dist * v - focus_dist * w;
    horizontal = 2.0f * half_width * focus_dist * u;
    vertical = 2.0f * half_height * focus_dist * v;
  }
  __device__ Ray get_ray(float s, float t, curandState* local_rand_state) {
    Vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
    Vec3 offset = u * rd.x() + v * rd.y();
    return Ray(origin + offset, lower_left_corner + s * horizontal +
                                    t * vertical - origin - offset, random_float(local_rand_state));
  }

  Vec3 origin;
  Vec3 lower_left_corner;
  Vec3 horizontal;
  Vec3 vertical;
  Vec3 u, v, w;
  float lens_radius;
};
#endif