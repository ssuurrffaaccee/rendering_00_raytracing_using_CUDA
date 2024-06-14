#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H
#include "hittable.hpp"
#include "util.hpp"
class HittablePtrVector {
 public:
  __device__ HittablePtrVector() {
    int pre_num = 400;
    data_ = (Hittable**)malloc(pre_num * sizeof(Hittable*));
    cap_ = pre_num;
    size_ = 0;
  }
  __device__ ~HittablePtrVector() { free(data_); }
  __device__ void push_back(Hittable* ptr) {
    if (size_ == cap_) {
      expand();
    }
    data_[size_] = ptr;
    size_++;
  }
  int size() { return size_; }
  Hittable** data_;
  int cap_;
  int size_;

 private:
  __device__ void expand() {
    int new_cap = 2 * cap_;
    Hittable** old_data = data_;
    data_ = (Hittable**)malloc(new_cap * sizeof(Hittable*));
    cap_ = new_cap;
    for (int i = 0; i < size_; i++) {
      data_[i] = old_data[i];
    }
    free(old_data);
  }
};

class HittableList : public Hittable {
 public:
  HittablePtrVector objects_;

  __device__ HittableList() {}
  __device__ HittableList(Hittable* object) { add(object); }

  __device__ void clear() {}

  __device__ void add(Hittable* object) {
    objects_.push_back(object);
    bbox_ = merge(bbox_, object->bounding_box());
  }
  __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec,
                      curandState* local_rand_state) const override {
    HitRecord temp_rec;
    auto hit_anything = false;
    auto closest_so_far = ray_t.max;
    for (int i = 0; i < objects_.size_; i++) {
      auto object = objects_.data_[i];
      if (object->hit(r, Interval{ray_t.min, closest_so_far}, temp_rec,
                      local_rand_state)) {
        hit_anything = true;
        closest_so_far = temp_rec.t_;
        rec = temp_rec;
      }
    }
    return hit_anything;
  }
  __device__ AABB bounding_box() const override { return bbox_; }
  __device__ float pdf_from(const Point3& origin, const Vec3& sample_direction,
                             curandState* local_rand_state) const override {
    auto weight = 1.0 / objects_.size_;
    auto sum = 0.0;

    for (int i = 0; i < objects_.size_; i++) {
      auto object = objects_.data_[i];
      sum += weight * object->pdf_from(origin, sample_direction, local_rand_state);
    }
    return sum;
  }
  __device__ Vec3 sample_from(const Vec3& origin,
                              curandState* local_rand_state) const override {
    auto int_size = int(objects_.size_);
    return objects_.data_[random_int(0, int_size - 1, local_rand_state)]->sample_from(
        origin, local_rand_state);
  }
};
#endif