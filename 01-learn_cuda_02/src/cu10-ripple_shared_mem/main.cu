#include "cuda_helper.hpp"
#include "common/viewerCPU.hpp"
#include "common/random.hpp"
#include <cmath>
struct DeferCudaFree
{
    DeferCudaFree(const std::function<void()>& func) : func_{func}
    {
    }
    ~DeferCudaFree()
    {
        if (func_ != nullptr)
        {
            func_();
        }
    }
    std::function<void()> func_;
};

#define DIM 1024
#define PI 3.1415926535897932f

__global__ void kernel( unsigned char *ptr,float period) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    __shared__ float    shared[16][16];

    // // now calculate the value at that position
    // const float period = 128.0f;

    shared[threadIdx.x][threadIdx.y] =
            255 * (sinf(x*2.0f*PI/ period) + 1.0f) *
                  (sinf(y*2.0f*PI/ period) + 1.0f) / 4.0f;

    // removing this syncthreads shows graphically what happens
    // when it doesn't exist.  this is an example of why we need it.
    __syncthreads();

    ptr[offset*4 + 0] = 0;
    ptr[offset*4 + 1] = shared[15-threadIdx.x][15-threadIdx.y];
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;
}
int
main()
{
    try
    {
        int width{1024};
        int height{1024};
        auto &cpuViewer = CPUViewer::getInstance();
        cpuViewer.init(width, height);
        auto pixelPtr = cpuViewer.getPixelsPtr();
        auto pixelDataSize = cpuViewer.getPixelsDataSize();
        unsigned char *tempPixel_gpu;
        HANDLE_ERROR(cudaMalloc((void **)&tempPixel_gpu, pixelDataSize));
        DeferCudaFree deferFree([tempPixel_gpu](){
            HANDLE_ERROR(cudaFree(tempPixel_gpu));
        });
        dim3    blocks(width/16,height/16);
        dim3    threads(16,16);
        float period{1.0f};
        auto updateFunc = [period, pixelDataSize, blocks,threads, tempPixel_gpu](unsigned char *pixelPtr, int width, int height) mutable
        {
            kernel<<<blocks, threads>>>(tempPixel_gpu,period);
            HANDLE_ERROR(cudaMemcpy(pixelPtr, tempPixel_gpu,
                                    pixelDataSize,
                                    cudaMemcpyDeviceToHost));
            period = std::fmod(period+0.01,18.0f);
        };
        cpuViewer.displayAndExit(updateFunc);
    }
    catch (MyExceptoin &e)
    {
        std::cout << e.what() << "\n";
    }
    catch (std::exception &e)
    {
        std::cout << e.what() << "\n";
    }
    return 0;
}