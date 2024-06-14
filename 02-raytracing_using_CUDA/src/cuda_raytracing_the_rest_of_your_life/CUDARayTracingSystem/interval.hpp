#ifndef INTERVAL_H
#define INTERVAL_H
#include "util.hpp"
class Interval {
 public:
  float min, max;

  __device__ Interval()
      : min(+INFINITY_), max(-INFINITY_) {}  // Default Interval is empty

  __device__ Interval(float _min, float _max) : min(_min), max(_max) {}

  __device__ float size() const { return max - min; }

  __device__ Interval expand(float delta) const {
    auto padding = delta / 2;
    return Interval(min - padding, max + padding);
  }

  __device__ bool contains(float x) const { return min <= x && x <= max; }

  __device__ bool surrounds(float x) const { return min < x && x < max; }

  __device__ float clamp(float x) const {
    if (x < min) return min;
    if (x > max) return max;
    return x;
  }

  static Interval empty, universe;
};
__device__ Interval merge(const Interval& a, const Interval& b) {
  return Interval{mymin(a.min, b.min), mymax(a.max, b.max)};
}
__device__ Interval operator+(const Interval& ival, float displacement) {
  return Interval(ival.min + displacement, ival.max + displacement);
}

__device__ Interval operator+(float displacement, const Interval& ival) {
  return ival + displacement;
}
// __device__ Interval Interval_empty = Interval(+INFINITY_, -INFINITY_);
// __device__ Interval Interval_universe = Interval(-INFINITY_, +INFINITY_);

#endif