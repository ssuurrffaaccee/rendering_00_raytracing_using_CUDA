#include "cuda_helper.hpp"
#include "common/viewerCPU.hpp"
#include "common/random.hpp"
#include <cmath>

struct DeferCudaFree
{
    DeferCudaFree(const std::function<void()> &func) : func_{func}
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

// #define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f

// // these exist on the GPU side
// texture<float> texConstSrc;
// texture<float> texIn;
// texture<float> texOut;
// this kernel takes in a 2-d array of floats
// it updates the value-of-interest by a scaled value based
// on itself and its nearest neighbors
__global__ void blend_kernel(float *texIn, float *texOut, float *dst,
                             bool dstOut, int DIM)
{
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int left = offset - 1;
    int right = offset + 1;
    if (x == 0)
        left++;
    if (x == DIM - 1)
        right--;

    int top = offset - DIM;
    int bottom = offset + DIM;
    if (y == 0)
        top += DIM;
    if (y == DIM - 1)
        bottom -= DIM;

    float t, l, c, r, b;
    if (dstOut)
    {
        t = texIn[top];
        l = texIn[left];
        c = texIn[offset];
        r = texIn[right];
        b = texIn[bottom];
    }
    else
    {
        t = texOut[top];
        l = texOut[left];
        c = texOut[offset];
        r = texOut[right];
        b = texOut[bottom];
    }
    dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}

// NOTE - texOffsetConstSrc could either be passed as a
// parameter to this function, or passed in __constant__ memory
// if we declared it as a global above, it would be
// a parameter here:
// __global__ void copy_const_kernel( float *iptr,
//                                    size_t texOffset )
__global__ void copy_const_kernel(float *texConstSrc, float *iptr)
{
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float c = texConstSrc[offset];
    if (c != 0)
        iptr[offset] = c;
}
__device__ unsigned char value(float n1, float n2, int hue)
{
    if (hue > 360)
        hue -= 360;
    else if (hue < 0)
        hue += 360;

    if (hue < 60)
        return (unsigned char)(255 * (n1 + (n2 - n1) * hue / 60));
    if (hue < 180)
        return (unsigned char)(255 * n2);
    if (hue < 240)
        return (unsigned char)(255 * (n1 + (n2 - n1) * (240 - hue) / 60));
    return (unsigned char)(255 * n1);
}
__global__ void float_to_color(unsigned char *optr,
                               const float *outSrc)
{
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float l = outSrc[offset];
    float s = 1;
    int h = (180 + (int)(360.0f * outSrc[offset])) % 360;
    float m1, m2;

    if (l <= 0.5f)
        m2 = l * (1 + s);
    else
        m2 = l + s - l * s;
    m1 = 2 * l - m2;

    optr[offset * 4 + 0] = value(m1, m2, h + 120);
    optr[offset * 4 + 1] = value(m1, m2, h);
    optr[offset * 4 + 2] = value(m1, m2, h - 120);
    optr[offset * 4 + 3] = 255;
}

int main()
{
    try
    {
        int DIM{1024};
        int width{DIM};
        int height{DIM};
        auto &cpuViewer = CPUViewer::getInstance();
        cpuViewer.init(width, height);
        auto pixelPtr = cpuViewer.getPixelsPtr();
        auto pixelDataSize = cpuViewer.getPixelsDataSize();
        // capture the start time
        cudaEvent_t start, stop;
        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));
        HANDLE_ERROR(cudaEventRecord(start, 0));
        DeferCudaFree eventDefer([start, stop]()
                                 {
            HANDLE_ERROR( cudaEventDestroy( start ) );
            HANDLE_ERROR( cudaEventDestroy( stop ) ); });
        float *dev_inSrc;
        float *dev_outSrc;
        float *dev_constSrc;
        // assume float == 4 chars in size (ie rgba)
        HANDLE_ERROR(cudaMalloc((void **)&dev_inSrc,
                                pixelDataSize));
        HANDLE_ERROR(cudaMalloc((void **)&dev_outSrc,
                                pixelDataSize));
        HANDLE_ERROR(cudaMalloc((void **)&dev_constSrc,
                                pixelDataSize));
        DeferCudaFree defferFreeAll([=](){
             HANDLE_ERROR(cudaFree(dev_inSrc));
             HANDLE_ERROR(cudaFree(dev_outSrc));
             HANDLE_ERROR(cudaFree(dev_constSrc));
        });
        {
            // intialize the constant data
            float *temp = (float *)malloc(pixelDataSize);
            for (int i = 0; i < DIM * DIM; i++)
            {
                temp[i] = 0;
                int x = i % DIM;
                int y = i / DIM;
                if ((x > 300) && (x < 600) && (y > 310) && (y < 601))
                    temp[i] = MAX_TEMP;
            }
            temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
            temp[DIM * 700 + 100] = MIN_TEMP;
            temp[DIM * 300 + 300] = MIN_TEMP;
            temp[DIM * 200 + 700] = MIN_TEMP;
            for (int y = 800; y < 900; y++)
            {
                for (int x = 400; x < 500; x++)
                {
                    temp[x + y * DIM] = MIN_TEMP;
                }
            }
            HANDLE_ERROR(cudaMemcpy(dev_constSrc, temp,
                                    pixelDataSize,
                                    cudaMemcpyHostToDevice));
            // initialize the input data
            for (int y = 800; y < DIM; y++)
            {
                for (int x = 0; x < 200; x++)
                {
                    temp[x + y * DIM] = MAX_TEMP;
                }
            }
            HANDLE_ERROR(cudaMemcpy(dev_inSrc, temp,
                                    pixelDataSize,
                                    cudaMemcpyHostToDevice));
            free(temp);
        }
        unsigned char *tempPixel_gpu;
        HANDLE_ERROR(cudaMalloc((void **)&tempPixel_gpu, pixelDataSize));
        DeferCudaFree deferFree([tempPixel_gpu]()
                                { HANDLE_ERROR(cudaFree(tempPixel_gpu)); });
        float totalTime{0.0};
        int frames{0};
        auto UpdateFunc = [&](unsigned char *pixelPtr, int width, int height) mutable
        {
            HANDLE_ERROR(cudaEventRecord(start, 0));
            dim3 blocks(DIM / 16, DIM / 16);
            dim3 threads(16, 16);

            // since tex is global and bound, we have to use a flag to
            // select which is in/out per iteration
            volatile bool dstOut = true;
            for (int i = 0; i < 90; i++)
            {
                float *in, *out;
                if (dstOut)
                {
                    in = dev_inSrc;
                    out = dev_outSrc;
                }
                else
                {
                    out = dev_inSrc;
                    in = dev_outSrc;
                }
                copy_const_kernel<<<blocks, threads>>>(dev_constSrc, in);
                blend_kernel<<<blocks, threads>>>(dev_inSrc, dev_outSrc, out, dstOut, DIM);
                dstOut = !dstOut;
            }
            float_to_color<<<blocks, threads>>>(tempPixel_gpu,
                                                dev_inSrc);

            HANDLE_ERROR(cudaMemcpy(pixelPtr,
                                    tempPixel_gpu,
                                    pixelDataSize,
                                    cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaEventRecord(stop, 0));
            HANDLE_ERROR(cudaEventSynchronize(stop));
            float elapsedTime;
            HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
                                              start, stop));
            totalTime += elapsedTime;
            ++frames;
            printf("Average Time per frame:  %3.1f ms\n",
                   totalTime / frames);
        };
        cpuViewer.displayAndExit(UpdateFunc);
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