#ifndef LOAD_TEXTURE_H
#define LOAD_TEXTURE_H
#include <string>
#include <functional>
void load_texture_from_file(const std::string &file, bool alpha, bool flip_vertically, const std::function<void(unsigned char*,int,int,int)>& func);
#endif