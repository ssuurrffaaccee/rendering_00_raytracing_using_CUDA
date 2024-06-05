#ifndef CUDA_ERROR_H
#define CUDA_ERROR_H
#include <stdio.h>
#include <cuda_runtime_api.h>

static void HandleError(cudaError_t err,
                        const char *file,
                        int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#define HANDLE_NULL(a)                                      \
    {                                                       \
        if (a == NULL)                                      \
        {                                                   \
            printf("Host memory failed in %s at line %d\n", \
                   __FILE__, __LINE__);                     \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    }

// class Texture
// {
// public:
//     Texture(Texture &&) = delete;
//     Texture(Texture &) = delete;
//     Texture &operator=(Texture &) = delete;
//     Texture &operator=(Texture &&) = delete;
//     Texture(float *dataOnCPU, int width, int height)
//     {
//         int size = width * height * sizeof(float);
//         // Allocate array and copy image data
//         cudaChannelFormatDesc channelDesc =
//             cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
//         cudaArray *cuArray;
//         HANDLE_ERROR(cudaMallocArray(&cuArray, &channelDesc, width, height));
//         HANDLE_ERROR(
//             cudaMemcpyToArray(cuArray, 0, 0, dataOnCPU, size, cudaMemcpyHostToDevice));
//         cudaResourceDesc texRes;
//         memset(&texRes, 0, sizeof(cudaResourceDesc));

//         texRes.resType = cudaResourceTypeArray;
//         texRes.res.array.array = cuArray;

//         cudaTextureDesc texDescr;
//         memset(&texDescr, 0, sizeof(cudaTextureDesc));

//         texDescr.normalizedCoords = true;
//         texDescr.filterMode = cudaFilterModeLinear;
//         texDescr.addressMode[0] = cudaAddressModeWrap;
//         texDescr.addressMode[1] = cudaAddressModeWrap;
//         texDescr.readMode = cudaReadModeElementType;
//         HANDLE_ERROR(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));
//     }
//     ~Texture()
//     {
//         HANDLE_ERROR(cudaDestroyTextureObject(tex));
//         HANDLE_ERROR(cudaFreeArray(cuArray));
//     }
//     cudaTextureObject_t tex;

// private:
//     cudaArray *cuArray{nullptr};
// };
#endif