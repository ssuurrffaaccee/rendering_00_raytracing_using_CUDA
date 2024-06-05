#pragma once
#include "common.h"
struct Transform3D {
  glm::vec3 position_{0.0f, 0.0f, 0.0f};
  glm::vec3 scale_{1.0f, 1.0f, 1.0f};
  glm::vec3 rotation_{0.0f, 0.0, 0.0f};
  glm::mat4 getModelTranstorm() {
    auto model = glm::mat4(1.0f);
    model = glm::translate(model, position_);
    model = glm::scale(model, scale_);
    model = glm::rotate(model, rotation_[2], glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::rotate(model, rotation_[1], glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, rotation_[0], glm::vec3(1.0f, 0.0f, 0.0f));
    return model;
  }
  glm::mat4 modelTranform_{1.0};
};