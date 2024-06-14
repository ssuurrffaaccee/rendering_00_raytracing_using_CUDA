#pragma once
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

#include "texture.h"
struct Vertex {
  // position
  glm::vec3 position_;
  // normal
  glm::vec3 normal_;
  // texCoords
  glm::vec2 texCoords_;
  // tangent
  glm::vec3 tangent_;
  // bitangent
  glm::vec3 bitangent_;
};
struct Mesh {
  // mesh Data
  std::vector<Vertex> vertices_;
  std::vector<unsigned int> indices_;
  uint32_t VAO_{0};
};
struct SimpleMesh{
  uint32_t VAO_{0};
  uint32_t indexSize{0};
};
struct Model {
  std::vector<std::shared_ptr<Mesh>> meshs_;
  // void clear();
};
Model loadModel(const std::string& directory, const std::string& names);