#ifndef KERNEL_H
#define KERNEL_H
#include "common/cuda_helper.hpp"

__global__ void kernel(uchar4* pixels,int width,int height,int tick);

void kernel_call(uchar4* pixels,int width,int height,int tick);
#endif