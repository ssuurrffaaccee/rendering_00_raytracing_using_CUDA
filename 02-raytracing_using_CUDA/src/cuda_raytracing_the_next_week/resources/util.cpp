
#include "util.h"
#include "resourceManager.h"
#include "util/check.h"
#include "util/fs.hpp"
#include <stb_image.h>
#include <glad/glad.h>
Texture2D loadTextureFromFile(const std::string &file) {
  // create texture object
  Texture2D texture;
  // load image
  int width, height, nrChannels;
  stbi_set_flip_vertically_on_load(true); 
  unsigned char *data = stbi_load(file.data(), &width, &height, &nrChannels, 0);
  CHECK(data != nullptr);
  GLenum format;
  if (nrChannels == 1) {
    texture.imageFormat = GL_RED;
    texture.internalFormat = GL_RED;
  } else if (nrChannels == 3) {
    texture.imageFormat = GL_RGB;
    texture.internalFormat = GL_RGB;
  } else if (nrChannels == 4) {
    texture.imageFormat = GL_RGBA;
    texture.internalFormat = GL_RGBA;
  }
  // now generate texture
  texture.generate(width, height, data);
  // and finally free image data
  stbi_image_free(data);
  return texture;
}