#include "cuda_helper.hpp"
#include "common/viewerCPU.hpp"
#include "common/random.hpp"

struct cuComplex
{
    float r;
    float i;
    __device__ cuComplex(float a, float b) : r(a), i(b) {}
    __device__ float magnitude2(void)
    {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex &a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    __device__ cuComplex operator+(const cuComplex &a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ int julia(int x, int y, int width, int height, float scale)
{
    // const float scale = 1.5;
    float jx = scale * (float)(width / 2 - x) / (width / 2);
    float jy = scale * (float)(height / 2 - y) / (height / 2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i = 0; i < 200; i++)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

__global__ void kernel(unsigned char *ptr, int width, int height, float scale)
{
    // map from blockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    // now calculate the value at that position
    int juliaValue = julia(x, y, width, height, scale);
    ptr[offset * 4 + 0] = 255 * juliaValue;
    ptr[offset * 4 + 1] = 0;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

template <typename T>
struct DeferCudaFree
{
    DeferCudaFree(T *data) : data_{data}
    {
    }
    ~DeferCudaFree()
    {
        if (data_ != nullptr)
        {
            HANDLE_ERROR(cudaFree(data_));
        }
    }
    T *data_ { nullptr };
};

int
main()
{
    try
    {
        int width{800};
        int height{800};
        auto &cpuViewer = CPUViewer::getInstance();
        cpuViewer.init(width, height);
        auto pixelPtr = cpuViewer.getPixelsPtr();
        auto pixelDataSize = cpuViewer.getPixelsDataSize();
        unsigned char *tempPixel_gpu;
        HANDLE_ERROR(cudaMalloc((void **)&tempPixel_gpu, pixelDataSize));
        DeferCudaFree deferFree(tempPixel_gpu);
        dim3 grid(width, height);
        float scale = 1.0f;
        auto updateFunc = [scale, pixelDataSize, grid, tempPixel_gpu](unsigned char *pixelPtr, int width, int height) mutable
        {
            kernel<<<grid, 1>>>(tempPixel_gpu, width, height, scale);
            scale *= 0.995;
            if (scale < 0.02f)
            {
                scale = 1.0f;
            }
            HANDLE_ERROR(cudaMemcpy(pixelPtr, tempPixel_gpu,
                                    pixelDataSize,
                                    cudaMemcpyDeviceToHost));
        };

        // std::cout << int(pixelPtr[0]) << " " << int(pixelPtr[1]) << " "
        //           << int(pixelPtr[2]) << " " << int(pixelPtr[3]) << "\n";
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