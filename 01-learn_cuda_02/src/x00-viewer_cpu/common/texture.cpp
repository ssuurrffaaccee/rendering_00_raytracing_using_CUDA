#include "texture.hpp"

#include <glad/glad.h>

#include "check.hpp"
Texture2D::Texture2D()
    : width(0),
      height(0),
      internalFormat(GL_RGB),
      imageFormat(GL_RGB),
      wrap_S(GL_REPEAT),
      wrap_T(GL_REPEAT),
      filterMin(GL_LINEAR),
      filterMax(GL_LINEAR) {
}

void Texture2D::generate(unsigned int width, unsigned int height,
                         unsigned char* data) {
  this->width = width;
  this->height = height;
  glGenTextures(1, &this->ID);
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
