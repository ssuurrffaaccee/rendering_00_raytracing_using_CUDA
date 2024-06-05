#include "common/random.hpp"
#include "common/viewerGPU.hpp"
#include "kernel.hpp"
int main() {
  try {
    int width{1024};
    int height{1024};
    int isAnim{false};
    auto &gpuViewer = GPUViewer::getInstance();
    gpuViewer.init(width, height,isAnim);
    auto pixelDataSize = gpuViewer.getPixelsDataSize();
    int ticks{0};
    auto updateFunc = [&](uchar4 *pixelPtr, int width,
                                      int height) {
      kernel_call(pixelPtr,width,height,ticks);
      ticks++;
    };
    gpuViewer.displayAndExit(updateFunc);
  } catch (MyExceptoin &e) {
    std::cout << e.what() << "\n";
  } catch (std::exception &e) {
    std::cout << e.what() << "\n";
  }
  return 0;
}