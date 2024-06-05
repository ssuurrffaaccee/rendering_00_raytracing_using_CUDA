#pragma once
#include <cstdlib>

inline float myRandom() { return std::rand() / (RAND_MAX + 1.0); }

inline float myRandom(float min, float max) {
  return min + (max - min) * myRandom();
}