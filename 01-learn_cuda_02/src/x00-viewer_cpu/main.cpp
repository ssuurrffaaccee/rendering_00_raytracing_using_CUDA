#include "common/viewerCPU.hpp"
#include "common/random.hpp"

int main()
{
  try
  {
    int width{800};
    int height{600};
    auto &cpuViewer = CPUViewer::getInstance();
    cpuViewer.init(width, height);
    auto pixelPtr = cpuViewer.getPixelsPtr();
    auto pixelDataSize = cpuViewer.getPixelsDataSize();
    // for (int i = 0; i < pixelDataSize; i = i + 4)
    // {
    //   pixelPtr[i] = 0;
    //   pixelPtr[i + 1] = 255;
    //   pixelPtr[i + 2] = 0;
    //   pixelPtr[i + 3] = 255;
    // }
    auto updateFunc = [pixelDataSize](unsigned char *pixelPtr, int width, int height)
    {
      for (int i = 0; i < pixelDataSize; i = i + 4)
      {
        float r = my_random();
        float g = my_random();
        float b = my_random();
        pixelPtr[i] = (unsigned char)(r * 255.0f);
        pixelPtr[i + 1] = (unsigned char)(g * 255.0f);
        pixelPtr[i + 2] = (unsigned char)(b * 255.0f);
        pixelPtr[i + 3] = 255;
      }
    };
    cpuViewer.displayAndExit(updateFunc);
    std::cout << int(pixelPtr[0]) << " " << int(pixelPtr[1]) << " "
              << int(pixelPtr[2]) << "\n";
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