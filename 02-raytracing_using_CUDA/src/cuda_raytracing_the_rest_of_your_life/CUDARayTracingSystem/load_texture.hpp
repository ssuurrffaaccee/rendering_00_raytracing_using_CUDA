#ifndef LOAD_CUDA_TEXTURE_H
#define LOAD_CUDA_TEXTURE_H
#include <string>
#include <cuda_runtime_api.h>
#include "resources/load_texture.hpp"
std::pair<cudaTextureObject_t,cudaArray*> load_cuda_texture_from_file(const std::string &file, bool alpha);
#endif