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

#define DIM 1024

#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f

struct Sphere
{
    float r, b, g;
    float radius;
    float x, y, z;
    __device__ float hit(float ox, float oy, float *n)
    {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius)
        {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius);
            return dz + z;
        }
        return -INF;
    }
};
#define SPHERES 20

__global__ void kernel(Sphere *s, unsigned char *ptr, int width, int height)
{
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float ox = (x - width / 2);
    float oy = (y - height / 2);

    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for (int i = 0; i < SPHERES; i++)
    {
        float n;
        float t = s[i].hit(ox, oy, &n);
        if (t > maxz)
        {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t;
        }
    }

    ptr[offset * 4 + 0] = (int)(r * 255);
    ptr[offset * 4 + 1] = (int)(g * 255);
    ptr[offset * 4 + 2] = (int)(b * 255);
    ptr[offset * 4 + 3] = 255;
}

int main()
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
        DeferCudaFree deferFree([tempPixel_gpu]()
                                { HANDLE_ERROR(cudaFree(tempPixel_gpu)); });

        // capture the start time
        cudaEvent_t start, stop;
        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));
        HANDLE_ERROR(cudaEventRecord(start, 0));
        DeferCudaFree eventDefer([start, stop]()
                                 {
            HANDLE_ERROR( cudaEventDestroy( start ) );
            HANDLE_ERROR( cudaEventDestroy( stop ) ); });
        Sphere *s;
        {
            // allocate memory for the Sphere dataset
            HANDLE_ERROR(cudaMalloc((void **)&s,
                                    sizeof(Sphere) * SPHERES));
            // allocate temp memory, initialize it, copy to
            // memory on the GPU, then free our temp memory
            Sphere *tempSphere = (Sphere *)malloc(sizeof(Sphere) * SPHERES);
            for (int i = 0; i < SPHERES; i++)
            {
                tempSphere[i].r = rnd(1.0f);
                tempSphere[i].g = rnd(1.0f);
                tempSphere[i].b = rnd(1.0f);
                tempSphere[i].x = rnd(1000.0f) - 500;
                tempSphere[i].y = rnd(1000.0f) - 500;
                tempSphere[i].z = rnd(1000.0f) - 500;
                tempSphere[i].radius = rnd(100.0f) + 20;
            }
            HANDLE_ERROR(cudaMemcpy(s, tempSphere,
                                    sizeof(Sphere) * SPHERES,
                                    cudaMemcpyHostToDevice));
            free(tempSphere);
        }
        DeferCudaFree sphereDefer([s]()
                                  { HANDLE_ERROR(cudaFree(s)); });
        dim3 blocks(width / 16, height / 16);
        dim3 threads(16, 16);
        kernel<<<blocks, threads>>>(s, tempPixel_gpu, width, height);
        HANDLE_ERROR(cudaMemcpy(pixelPtr, tempPixel_gpu,
                                pixelDataSize,
                                cudaMemcpyDeviceToHost));
        // get stop time, and display the timing results
        HANDLE_ERROR(cudaEventRecord(stop, 0));
        HANDLE_ERROR(cudaEventSynchronize(stop));
        float elapsedTime;
        HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
                                          start, stop));
        printf("Time to generate:  %3.1f ms\n", elapsedTime);
        cpuViewer.displayAndExit();
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