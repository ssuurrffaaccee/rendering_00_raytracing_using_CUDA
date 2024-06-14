#include "load_texture.hpp"
#include "check_cuda.h"
#include "cuda_runtime.h"
std::pair<cudaTextureObject_t,cudaArray*> load_cuda_texture_from_file(const std::string &file, bool alpha)
{
    int width, height, nrChannels;
    cudaTextureObject_t tex;
    cudaArray *cu_array;
    load_texture_from_file(file.c_str(), true, true, [&](unsigned char *d, int w, int h, int c)
                           {
       width = w;
       height = h;
       unsigned int size = width * height * sizeof(unsigned);
       std::vector<unsigned> temp_date_vector;
       temp_date_vector.resize(width * height,0);
       unsigned char* temp_data_chars = (unsigned char*)temp_date_vector.data();
       for(int i = 0; i< w * h; i++){
          temp_data_chars[i*4] = d[i*3];
          temp_data_chars[i*4+1] = d[i*3+1];
          temp_data_chars[i*4+2] = d[i*3+2];
       }
        // Allocate array and copy image data
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        checkCudaErrors(cudaMallocArray(&cu_array, &channelDesc, width, height));
        checkCudaErrors(
            cudaMemcpyToArray(cu_array, 0, 0, temp_data_chars, size, cudaMemcpyHostToDevice));

        cudaResourceDesc texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));

        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array = cu_array;

        cudaTextureDesc texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));

        texDescr.normalizedCoords = true;
        texDescr.filterMode = cudaFilterModePoint;
        texDescr.addressMode[0] = cudaAddressModeClamp;
        texDescr.addressMode[1] = cudaAddressModeClamp;
        texDescr.readMode = cudaReadModeElementType;

        checkCudaErrors(cudaCreateTextureObject(&tex, &texRes, &texDescr, nullptr)); });
    return std::make_pair(tex,cu_array);
}