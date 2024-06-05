#include "common/random.hpp"
#include "common/viewerGPU.hpp"
#include "kernel.hpp"
int main() {
  try {
    int width{1024};
    int height{1024};
    int isAnim{true};
    auto &cpuViewer = CPUViewer::getInstance();
    cpuViewer.init(width, height, );
    auto pixelPtr = cpuViewer.getPixelsPtr();
    auto pixelDataSize = cpuViewer.getPixelsDataSize();
    int ticks{0};
    dim3 blocks(width / 16, height / 16, 1);
    dim3 threads(16, 16, 1);
    auto updateFunc = [pixelDataSize](unsigned char *pixelPtr, int width,
                                      int height) {
      kernel<<<blocks, threads>>>(pixelPtr, width, height, ticks);
      ticks++;
    };
    cpuViewer.displayAndExit(updateFunc);
    std::cout << int(pixelPtr[0]) << " " << int(pixelPtr[1]) << " "
              << int(pixelPtr[2]) << "\n";
  } catch (MyExceptoin &e) {
    std::cout << e.what() << "\n";
  } catch (std::exception &e) {
    std::cout << e.what() << "\n";
  }
  return 0;
}