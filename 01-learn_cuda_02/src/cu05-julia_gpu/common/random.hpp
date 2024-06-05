#ifndef RANDOM_H
#define RNADOM_H
#include <cstdlib>

inline float my_random() {
  return std::rand() / (RAND_MAX + 1.0);
}

inline float my_random(float min, float max) {
  return min + (max - min) * my_random();
}
#endif
