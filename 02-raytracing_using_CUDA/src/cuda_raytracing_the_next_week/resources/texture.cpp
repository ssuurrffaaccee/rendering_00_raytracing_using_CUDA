#include "texture.h"

#include <glad/glad.h>

#include "util/check.h"
Texture2D::Texture2D()
    : width(0),
      height(0),
      internalFormat(GL_RGB),
      imageFormat(GL_RGB),
      wrap_S(GL_REPEAT),
      wrap_T(GL_REPEAT),
      filterMin(GL_LINEAR),
      filterMax(GL_LINEAR) {
  glGenTextures(1, &this->ID);
}

void Texture2D::generate(unsigned int width, unsigned int height,
                         unsigned char* data) {
  this->width = width;
  this->height = height;
  // create Texture
  glBindTexture(GL_TEXTURE_2D, this->ID);
  glTexImage2D(GL_TEXTURE_2D, 0, this->internalFormat, width, height, 0,
               this->imageFormat, GL_UNSIGNED_BYTE, data);
  // set Texture wrap and filter modes
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, this->wrap_S);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, this->wrap_T);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, this->filterMin);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, this->filterMax);
  // unbind texture
  glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture2D::bind() const { glBindTexture(GL_TEXTURE_2D, this->ID); }

TextureCube::TextureCube()
    : width(0),
      height(0),
      internalFormat(GL_RGB),
      imageFormat(GL_RGB),
      wrap_S(GL_REPEAT),
      wrap_T(GL_REPEAT),
      wrap_R(GL_REPEAT),
      filterMin(GL_LINEAR),
      filterMax(GL_LINEAR) {
  glGenTextures(1, &this->ID);
}

void TextureCube::generate(unsigned int width, unsigned int height,
                         const std::vector<unsigned char*>& datas) {
  this->width = width;
  this->height = height;
  // create Texture
  glBindTexture(GL_TEXTURE_CUBE_MAP, this->ID);
  CHECK(datas.size() == 6);
  for (unsigned int i = 0; i < 6; i++) {
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, this->internalFormat,
                 width, height, 0, this->imageFormat, GL_UNSIGNED_BYTE,
                 datas[i]);
  }

  // set Texture wrap and filter modes
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, this->filterMax);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, this->filterMin);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, this->wrap_S);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, this->wrap_T);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, this->wrap_R);
  // unbind texture
  glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

void TextureCube::bind() const { glBindTexture(GL_TEXTURE_CUBE_MAP, this->ID); }