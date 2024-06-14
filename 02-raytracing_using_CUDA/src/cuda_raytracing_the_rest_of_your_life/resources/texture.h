#pragma once
#include <vector>

class Texture2D {
 public:
  unsigned int ID;
  // texture image dimensions
  unsigned int width, height;  // width and height of loaded image in pixels
  // texture Format
  unsigned int internalFormat;  // format of texture object
  unsigned int imageFormat;     // format of loaded image
  // texture configuration
  unsigned int wrap_S;     // wrapping mode on S axis
  unsigned int wrap_T;     // wrapping mode on T axis
  unsigned int filterMin;  // filtering mode if texture pixels < screen pixels
  unsigned int filterMax;  // filtering mode if texture pixels > screen pixels
  Texture2D();
  void generate(unsigned int width, unsigned int height, unsigned char* data);
  // binds the texture as the current active GL_TEXTURE_2D texture object
  void bind() const;
};

class TextureCube {
 public:
  unsigned int ID;
  // texture image dimensions
  unsigned int width, height;  // width and height of loaded image in pixels
  // texture Format
  unsigned int internalFormat;  // format of texture object
  unsigned int imageFormat;     // format of loaded image
  // texture configuration
  unsigned int wrap_S;     // wrapping mode on S axis
  unsigned int wrap_T;     // wrapping mode on T axis
  unsigned int wrap_R;     // wrapping mode on R axis
  unsigned int filterMin;  // filtering mode if texture pixels < screen pixels
  unsigned int filterMax;  // filtering mode if texture pixels > screen pixels
  TextureCube();
  void generate(unsigned int width, unsigned int height,
                const std::vector<unsigned char*>& datas);
  // binds the texture as the current active GL_TEXTURE_2D texture object
  void bind() const;
};