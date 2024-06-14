#pragma once
#include <glm/glm.hpp>
#include <optional>

enum class BaseColorType {
  UNKONW,
  FIXED,
  RNADOM,
};

struct BaseColor {
  BaseColorType type_{BaseColorType::UNKONW};
  std::optional<glm::vec3> baseColor_;
  glm::vec3 getColor();
};