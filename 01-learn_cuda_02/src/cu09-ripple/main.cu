#include "cuda_helper.hpp"
#include "common/viewerCPU.hpp"
#include "common/random.hpp"
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

__global__ void kernel( unsigned char *ptr, int ticks, int width,int height) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    // now calculate the value at that position
    float fx = x - width/2;
    float fy = y - height/2;
    float d = sqrtf( fx * fx + fy * fy );
    unsigned char grey = (unsigned char)(128.0f + 127.0f *
                                         cos(d/10.0f - ticks/7.0f) /
                                         (d/10.0f + 1.0f));    
    ptr[offset*4 + 0] = grey;
    ptr[offset*4 + 1] = grey;
    ptr[offset*4 + 2] = grey;
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
        int ticks{0};
        auto updateFunc = [ticks, pixelDataSize, blocks,threads, tempPixel_gpu](unsigned char *pixelPtr, int width, int height) mutable
        {
            kernel<<<blocks, threads>>>(tempPixel_gpu,ticks,width, height);
            HANDLE_ERROR(cudaMemcpy(pixelPtr, tempPixel_gpu,
                                    pixelDataSize,
                                    cudaMemcpyDeviceToHost));
            ticks++;
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