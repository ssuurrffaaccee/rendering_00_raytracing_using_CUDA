#include <cmath>

#include "kernel.hpp"

namespace {
float fract(float x) {
  static float _;
  return std::modf(x, _);
}

float hash(float t) {
  return fract(std::sin(t * 8.233f) * 43758.5453123) > 0.5f ? 0.0f : 1.0f;
}
}  // namespace
void kernel(unsigned char* pixels, int width, int height, int tick) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int pixelIndex = y * width + x;
  float r = hash(float(pixelIndex + tick));
  unsigned char gray = (unsigned char)(r * 255.0f);
  pixels[4 * pixelIndex] = gray;
  pixels[4 * pixelIndex + 1] = gray;
  pixels[4 * pixelIndex + 2] = gray;
  pixels[4 * pixelIndex + 3] = 255;
}