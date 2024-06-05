#pragma once 
#include "common.h"
struct Transform2D {
  glm::vec2 position_{0.0f, 0.0f};
  glm::vec2 size_{10.0f, 10.0f};
  float rotation_{0.0f};
};