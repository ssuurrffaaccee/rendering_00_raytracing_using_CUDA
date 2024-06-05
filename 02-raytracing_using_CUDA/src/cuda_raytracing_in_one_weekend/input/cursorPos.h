#pragma once 
#include <glm/glm.hpp>
struct CursorPos {
  glm::vec2 position_{0.0f, 0.0f};
  glm::vec2 positionOffset_{0.0f,0.0f};
  glm::vec2 scroll_{0.0f,0.0f};
};