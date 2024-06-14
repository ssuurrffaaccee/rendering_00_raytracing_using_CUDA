
#include <glad/glad.h>
// #define STB_IMAGE_IMPLEMENTATION
// #include <stb_image.h>
#include "load_texture.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include "resourceManager.h"
#include "simpleMesh.h"
#include "util/check.h"
// Instantiate static variables
std::map<std::string, Texture2D> ResourceManager::Textures;
std::map<std::string, TextureCube> ResourceManager::Textures_Cube;
std::map<std::string, Shader> ResourceManager::Shaders;
std::map<std::string, Model> ResourceManager::Models;
std::map<std::string, SimpleMesh> ResourceManager::SimpleMeshs;
std::map<std::string, Material> ResourceManager::Materials;
Shader ResourceManager::loadShader(const char *vShaderFile,
                                   const char *fShaderFile,
                                   const char *gShaderFile,
                                   const std::string &name) {
  Shaders[name] = loadShaderFromFile(vShaderFile, fShaderFile, gShaderFile);
  return Shaders[name];
}

Shader ResourceManager::getShader(const std::string &name) {
  auto iter = Shaders.find(name);
  CHECK(iter != Shaders.end());
  return iter->second;
}

Texture2D ResourceManager::loadTexture(const char *file, bool alpha,
                                       const std::string &name) {
  Textures[name] = loadTextureFromFile(file, alpha);
  return Textures[name];
}
TextureCube ResourceManager::loadTextureCube(
    const std::vector<std::string> &files, bool alpha,
    const std::string &name) {
  Textures_Cube[name] = loadTextureCubeFromFile(files, alpha);
  return Textures_Cube[name];
}
Texture2D ResourceManager::getTexture(const std::string &name) {
  auto iter = Textures.find(name);
  CHECK(iter != Textures.end());
  return iter->second;
}
TextureCube ResourceManager::getTextureCube(const std::string &name) {
  auto iter = Textures_Cube.find(name);
  CHECK(iter != Textures_Cube.end());
  return iter->second;
}
Model ResourceManager::loadModel(const std::string &name,
                                 const std::string &directory,
                                 const std::string &meshName) {
  auto model = ::loadModel(directory, meshName);
  Models[name] = model;
  return model;
}
Model ResourceManager::getModel(const std::string &name) {
  auto iter = Models.find(name);
  CHECK(iter != Models.end());
  return iter->second;
}

void ResourceManager::clear() {
  // (properly) delete all shaders
  for (auto iter : Shaders) glDeleteProgram(iter.second.ID);
  // (properly) delete all textures
  for (auto iter : Textures) glDeleteTextures(1, &iter.second.ID);
  // for (auto iter : Models) iter.second.clear();
  for (auto iter : Materials) iter.second.clear();
  // for(auto iter : SimpleMeshs)
}

Shader ResourceManager::loadShaderFromFile(const char *vShaderFile,
                                           const char *fShaderFile,
                                           const char *gShaderFile) {
  // 1. retrieve the vertex/fragment source code from filePath
  std::string vertexCode;
  std::string fragmentCode;
  std::string geometryCode;
  try {
    // open files
    std::ifstream vertexShaderFile(vShaderFile);
    std::ifstream fragmentShaderFile(fShaderFile);
    std::stringstream vShaderStream, fShaderStream;
    // read file's buffer contents into streams
    vShaderStream << vertexShaderFile.rdbuf();
    fShaderStream << fragmentShaderFile.rdbuf();
    // close file handlers
    vertexShaderFile.close();
    fragmentShaderFile.close();
    // convert stream into string
    vertexCode = vShaderStream.str();
    fragmentCode = fShaderStream.str();
    std::cout << "vert: " << vertexCode << "\n\n frag: " << fragmentCode
              << "\n";
    // if geometry shader path is present, also load a geometry shader
    if (gShaderFile != nullptr) {
      std::ifstream geometryShaderFile(gShaderFile);
      std::stringstream gShaderStream;
      gShaderStream << geometryShaderFile.rdbuf();
      geometryShaderFile.close();
      geometryCode = gShaderStream.str();
    }
  } catch (std::exception e) {
    std::cout << "ERROR::SHADER: Failed to read shader files" << std::endl;
  }
  const char *vShaderCode = vertexCode.c_str();
  const char *fShaderCode = fragmentCode.c_str();
  const char *gShaderCode = geometryCode.c_str();
  // 2. now create shader object from source code
  Shader shader;
  shader.compile(vShaderCode, fShaderCode,
                 gShaderFile != nullptr ? gShaderCode : nullptr);
  return shader;
}

Texture2D ResourceManager::loadTextureFromFile(const char *file, bool alpha) {
  // create texture object
  Texture2D texture;
  if (alpha) {
    texture.internalFormat = GL_RGBA;
    texture.imageFormat = GL_RGBA;
  }
  // load image
  int width, height, nrChannels;
  unsigned char* data;
  load_texture_from_file(file,true,true,[&](unsigned char* d,int w,int h,int channels){
      width = w;
      height = h;
      nrChannels = channels;
      data = d;
  });
  // now generate texture
  texture.generate(width, height, data);
  return texture;
}
TextureCube ResourceManager::loadTextureCubeFromFile(
    const std::vector<std::string> &files, bool alpha) {
  CHECK(files.size() == 6)
  // create texture object
  TextureCube texture;
  if (alpha) {
    texture.internalFormat = GL_RGBA;
    texture.imageFormat = GL_RGBA;
  }
  // load image
  int width, height, nrChannels;
  std::vector<unsigned char *> datas;
  for (int i = 0; i < files.size(); i++) {
    std::cout<<files[i]<<"\n";
    load_texture_from_file(files[i],true,false,[&](unsigned char* data,int w,int h,int channels){
        CHECK(data != nullptr);
        datas.push_back(data);
        width = w;
        height = h;
        nrChannels = channels;
    });
    // unsigned char *data =
    //     stbi_load(files[i].c_str(), &width, &height, &nrChannels, 0);
    // CHECK(data != nullptr);
    // datas.push_back(data);
  }
  // now generate texture
  texture.generate(width, height, datas);
  return texture;
}

void ResourceManager::loadAllSimpleMesh() {
  auto cube = loadCube();
  SimpleMeshs["cube"] = cube;
}
SimpleMesh ResourceManager::getSimpleMesh(const std::string &name) {
  auto iter = SimpleMeshs.find(name);
  CHECK(iter != SimpleMeshs.end())
  return iter->second;
}

void ResourceManager::loadMaterials(const std::string &name,
                                    const std::string &dir,
                                    const std::vector<std::string> &names) {
  Materials[name] = loadMaterial(dir, names);
}
Material ResourceManager::getMaterial(const std::string &name) {
  auto iter = Materials.find(name);
  CHECK(iter != Materials.end())
  return iter->second;
}