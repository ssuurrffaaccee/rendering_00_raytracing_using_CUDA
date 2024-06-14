#include "material.h"

#include <glad/glad.h>

#include "util.h"
#include "util/check.h"
#include "util/fs.hpp"
Material loadMaterial(const std::string &directory,
                      const std::vector<std::string> &names) {
  CHECK_WITH_INFO(names.size() == 5, std::to_string(names.size()))
  std::string absDirectory = fs::absolute(directory);
  std::string diffusePath = absDirectory + names[0];
  CHECK_WITH_INFO(fs::exist(diffusePath), diffusePath)
  std::string normalPath = absDirectory + names[1];
  CHECK_WITH_INFO(fs::exist(normalPath), normalPath)
  std::string roughnessPath = absDirectory + names[2];
  CHECK_WITH_INFO(fs::exist(roughnessPath), roughnessPath)
  std::string specularPath = absDirectory + names[3];
  CHECK_WITH_INFO(fs::exist(specularPath), specularPath)
  std::string aoPath = absDirectory + names[4];
  CHECK_WITH_INFO(fs::exist(aoPath), aoPath)
  Material material;
  material.ao_ = loadTextureFromFile(aoPath);
  material.diffuse_ = loadTextureFromFile(diffusePath);
  material.normal_ = loadTextureFromFile(normalPath);
  material.roughness_ = loadTextureFromFile(roughnessPath);
  material.specular_ = loadTextureFromFile(specularPath);
  return material;
}
void Material::clear() {
  glDeleteTextures(1, &ao_.ID);
  glDeleteTextures(1, &diffuse_.ID);
  glDeleteTextures(1, &normal_.ID);
  glDeleteTextures(1, &roughness_.ID);
  glDeleteTextures(1, &specular_.ID);
}