#pragma once
#include <string>
#include <vector>

#include "texture.h"
struct Material {
  Texture2D ao_;
  Texture2D diffuse_;
  Texture2D normal_;
  Texture2D roughness_;
  Texture2D specular_;
  void clear();
};
Material loadMaterial(const std::string &directory,
                      const std::vector<std::string> &names);