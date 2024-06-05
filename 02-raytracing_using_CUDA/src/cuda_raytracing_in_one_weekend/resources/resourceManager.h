#pragma once
#include <map>
#include <memory>
#include <string>

#include "material.h"
#include "model.h"
#include "shader.h"
#include "texture.h"
class ResourceManager {
 public:
  static std::map<std::string, Shader> Shaders;
  static std::map<std::string, Texture2D> Textures;
  static std::map<std::string, TextureCube> Textures_Cube;
  static std::map<std::string, Model> Models;
  static std::map<std::string, SimpleMesh> SimpleMeshs;
  static std::map<std::string, Material> Materials;
  static void loadMaterials(const std::string &name, const std::string &dir,
                            const std::vector<std::string> &names);
  static Material getMaterial(const std::string &name);
  static void loadAllSimpleMesh();
  static SimpleMesh getSimpleMesh(const std::string &name);
  static Shader loadShader(const char *vShaderFile, const char *fShaderFile,
                           const char *gShaderFile, const std::string &name);
  static Shader getShader(const std::string &name);
  static Texture2D loadTexture(const char *file, bool alpha,
                               const std::string &name);
  static TextureCube loadTextureCube(
      const std::vector<std::string> &files,bool alpha,
      const std::string &name);
  static Texture2D getTexture(const std::string &name);
  static TextureCube getTextureCube(const std::string &name);
  static Model loadModel(const std::string &name, const std::string &directory,
                         const std::string &meshName);
  static Model getModel(const std::string &name);
  static void clear();
  static Texture2D loadTextureFromFile(const char *file);

 private:
  ResourceManager() {}
  static Shader loadShaderFromFile(const char *vShaderFile,
                                   const char *fShaderFile,
                                   const char *gShaderFile = nullptr);
  static Texture2D loadTextureFromFile(const char *file, bool alpha);
  static TextureCube loadTextureCubeFromFile(
      const std::vector<std::string> &files, bool alpha);
};