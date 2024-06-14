#ifndef BVH_NODE_H
#define BVH_NODE_H
#include <algorithm>
// #include <stack>

#include <thrust/sort.h>

#include "hittable.hpp"
#include "hittable_list.hpp"
#include "memory.hpp"
#include "random.hpp"
template <typename T>
struct Stack {
  __device__ Stack(int cap) : cap_{cap}, size_{0} {
    data_ = (T*)malloc(cap_ * sizeof(T));
  }
  __device__ void push(const T& v) {
    data_[size_] = v;
    size_++;
  }
  __device__ ~Stack() { free(data_); }
  __device__ T& top() { return data_[size_ - 1]; }
  __device__ void pop() { size_--; }
  __device__ bool empty() { return size_ == 0; }

 private:
  T* data_{nullptr};
  int cap_;
  int size_;
};

template <typename T, size_t N>
struct StaticStack {
  __device__ StaticStack() : cap_{N}, size_{0} {}
  __device__ void push(const T& v) {
    data_[size_] = v;
    size_++;
  }
  __device__ ~StaticStack() {}
  __device__ T& top() { return data_[size_ - 1]; }
  __device__ void pop() { size_--; }
  __device__ bool empty() { return size_ == 0; }

 private:
  T data_[N];
  int cap_;
  int size_;
};

// __device__ Hittable* make_bvh_node_internal_(Hittable** objects, int size,
//                                              curandState* local_rand_state);
class BVHNode : public Hittable {
 public:
  __device__ BVHNode() { is_bvh_node_ = true; }
  __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec,
                      curandState* local_rand_state) const override {
    StaticStack<Hittable*,20> stack;
    HitRecord closest_rec;
    HitRecord temp_rec;
    Interval hit_interval = ray_t;
    closest_rec.t_ = +INFINITY_;
    temp_rec.t_ = +INFINITY_;
    bool is_hit{false};
    Hittable* root = (Hittable*)this;
    stack.push(root);
    while (!stack.empty()) {
      Hittable* cur = stack.top();
      stack.pop();
      if (cur->bbox_.hit(r, hit_interval)) {
        if (cur->is_bvh_node_) {
          BVHNode* bvh_node = (BVHNode*)cur;
          stack.push(bvh_node->left_);
          stack.push(bvh_node->right_);
        } else {
          bool is_cur_hit = cur->hit(r, hit_interval, temp_rec, local_rand_state);
          if (is_cur_hit) {
            is_hit = true;
            hit_interval = Interval(hit_interval.min, temp_rec.t_);
            if (temp_rec.t_ >= 0.0f && temp_rec.t_ < closest_rec.t_) {
              closest_rec = temp_rec;
            }
          }
        }
      }
    }
    rec = closest_rec;
    return is_hit;
  }
  __device__ AABB bounding_box() const override { return bbox_; }
  __device__ friend Hittable* make_bvh_node_internal_(
      Hittable** objects, int size, curandState* local_rand_state);

 private:
  Hittable* left_;   // BVHNode or other
  Hittable* right_;  // BVHNode or other
  // EMBEDDING_STACK(Hittable*, 10)
};
__device__ bool box_compare(Hittable* a, Hittable* b, int axis_index) {
  return a->bounding_box().axis(axis_index).min <
         b->bounding_box().axis(axis_index).min;
}

__device__ bool box_x_compare(Hittable* a, Hittable* b) {
  return box_compare(a, b, 0);
}

__device__ bool box_y_compare(Hittable* a, Hittable* b) {
  return box_compare(a, b, 1);
}

__device__ bool box_z_compare(Hittable* a, Hittable* b) {
  return box_compare(a, b, 2);
}
__device__ Hittable* make_bvh_node(HittableList* world,
                                   curandState* local_rand_state) {
  return make_bvh_node_internal_(world->objects_.data_, world->objects_.size_,
                                 local_rand_state);
}
enum class SourceType {
  FROM_PARENT,
  FROM_LEFT,
  FROM_RIGHT,
};

struct ControlNode {
  __device__ ControlNode(BVHNode* node, SourceType source_type, int start,
                         int end)
      : node_{node}, source_type_{source_type}, start_{start}, end_{end} {}
  BVHNode* node_;
  SourceType source_type_;
  int start_{0};
  int end_{0};
};
__device__ Hittable* make_bvh_node_internal_(Hittable** objects, int size,
                                             curandState* local_rand_state) {
  Stack<ControlNode> s(1000);
  BVHNode* root = new BVHNode();
  memory_recorder_for_hittable->record(root);
  s.push(ControlNode{root, SourceType::FROM_PARENT, 0, size});
  while (!s.empty()) {
    auto cur = s.top();
    s.pop();
    if (cur.source_type_ == SourceType::FROM_PARENT) {
      int object_span = cur.end_ - cur.start_;
      if (object_span == 1) {
        cur.node_->left_ = objects[cur.start_];
        cur.node_->right_ = objects[cur.start_];
        cur.node_->bbox_ = merge(cur.node_->left_->bounding_box(),
                                 cur.node_->right_->bounding_box());
      } else {
        int axis = random_int(0, 2, local_rand_state);
        auto comparator = (axis == 0)   ? box_x_compare
                          : (axis == 1) ? box_y_compare
                                        : box_z_compare;
        if (object_span == 2) {
          cur.node_->left_ = objects[cur.start_];
          cur.node_->right_ = objects[cur.start_ + 1];
          cur.node_->bbox_ = merge(cur.node_->left_->bounding_box(),
                                   cur.node_->right_->bounding_box());
        } else {
          thrust::sort(objects + cur.start_, objects + cur.end_, comparator);
          auto mid = cur.start_ + object_span / 2;
          cur.node_->left_ = new BVHNode();
          cur.node_->right_ = new BVHNode();
          memory_recorder_for_hittable->record(cur.node_->left_);
          memory_recorder_for_hittable->record(cur.node_->right_);
          cur.source_type_ = SourceType::FROM_LEFT;
          s.push(cur);
          s.push(ControlNode{
              (BVHNode*)cur.node_->left_,
              SourceType::FROM_PARENT,
              cur.start_,
              mid,
          });
        }
      }
      continue;
    }
    if (cur.source_type_ == SourceType::FROM_LEFT) {
      auto mid = cur.start_ + (cur.end_ - cur.start_) / 2;
      cur.source_type_ = SourceType::FROM_RIGHT;
      s.push(cur);
      s.push(ControlNode{
          (BVHNode*)cur.node_->right_,
          SourceType::FROM_PARENT,
          mid,
          cur.end_,
      });
      continue;
    }
    if (cur.source_type_ == SourceType::FROM_RIGHT) {
      cur.node_->bbox_ = merge(cur.node_->left_->bounding_box(),
                               cur.node_->right_->bounding_box());
      continue;
    }
  }
  return root;
}
#endif