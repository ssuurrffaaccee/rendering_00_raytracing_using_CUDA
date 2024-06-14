#include "load_texture.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <iostream>
void load_texture_from_file(const std::string &file, bool alpha,bool flip_vertically, const std::function<void(unsigned char*,int,int,int)>& func){
       int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(flip_vertically);
    unsigned char *data = stbi_load(file.data(), &width, &height, &nrChannels, 0);
    std::cout<<width<<" "<<height<<" "<<nrChannels<<"\n";
    func(data,width,height,nrChannels);
    // and finally free image data
    stbi_image_free(data);
}