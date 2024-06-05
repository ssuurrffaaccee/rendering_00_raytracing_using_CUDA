#include "viewerGPU.hpp"

#include "cuda_helper.hpp"
static void glfwErrorPrint(int code, const char *info)
{
  std::cout << "GLFW error! code: " << code << " info: " << info << "\n";
}
void GPUViewer::init(int width, int height, bool isAnim)
{
  width_ = width;
  height_ = height;
  isAnim_ = isAnim;
  HANDLE_ERROR(cudaMalloc((void **)&pixels_, width_ * height * 4));
  // cudaGLSetGLDevice(0);
}
void GPUViewer::displayAndExit(
    std::function<void(uchar4 *, int, int)> updateFunc)
{
  initWindowAndOpengGL();
  registeCallback();
  buildeShader();
  initTexture();
  if (!isAnim_)
  {
    cudaArray_t texturePtr;
    HANDLE_ERROR(cudaGraphicsMapResources(1, &resource_, nullptr));
    HANDLE_ERROR(cudaGraphicsSubResourceGetMappedArray(&texturePtr, resource_, 0, 0));
    updateFunc(pixels_, width_, height_);
    HANDLE_ERROR(cudaMemcpyToArray(texturePtr, 0, 0, pixels_,
                                   width_ * height_ * 4, cudaMemcpyDeviceToDevice));
    HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource_, nullptr));
    while (!glfwWindowShouldClose(window_))
    {
      glfwPollEvents();
      draw();
      glfwSwapBuffers(window_);
    }
  }
  else
  {
    cudaArray_t texturePtr;
    while (!glfwWindowShouldClose(window_))
    {
      glfwPollEvents();
      HANDLE_ERROR(cudaGraphicsMapResources(1, &resource_, nullptr));
      HANDLE_ERROR(cudaGraphicsSubResourceGetMappedArray(&texturePtr, resource_, 0, 0));
      updateFunc(pixels_, width_, height_);
      HANDLE_ERROR(cudaMemcpyToArray(texturePtr, 0, 0, pixels_,
                                     width_ * height_ * 4, cudaMemcpyDeviceToDevice));
      HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource_, nullptr));
      draw();
      glfwSwapBuffers(window_);
    }
  }
}
GPUViewer::~GPUViewer()
{
  HANDLE_ERROR(cudaGraphicsUnregisterResource(resource_));
  HANDLE_ERROR(cudaFree(pixels_));
  glDeleteTextures(1, &textureToShow_.ID);
  glfwDestroyWindow(window_);
  glfwTerminate();
}
void GPUViewer::initWindowAndOpengGL()
{
  glfwSetErrorCallback(&glfwErrorPrint);
  CHECK_WITH_INFO(glfwInit() == GLFW_TRUE, "Failed to initialize GLFW");
  // glGetError();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  // #ifdef __APPLE__
  //     glfwWindowHint(GLFW_SCALE_FRAMEBUFFER, GLFW_FALSE);
  //     glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  // #endif
  glfwWindowHint(GLFW_RESIZABLE, false);
  // glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  window_ =
      glfwCreateWindow(width_, height_, "Window For Debug", nullptr, nullptr);
  CHECK_WITH_INFO_THEN(window_ != nullptr, "Failed to create window",
                       glfwTerminate(););

  // glad: load all OpenGL function pointers
  // ---------------------------------------
  glfwMakeContextCurrent(window_);
  CHECK_WITH_INFO(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress),
                  "Failed to initialize GLAD");
}
void GPUViewer::registeCallback()
{
  glfwSetWindowUserPointer(window_, this);
  glfwSetKeyCallback(window_, keyCallback);
}
void GPUViewer::keyCallback(GLFWwindow *window, int key, int scancode,
                            int action, int mods)
{
  // GPUViewer* app = (GPUViewer*)glfwGetWindowUserPointer(window);
  if (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE)
  {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  }
}
void GPUViewer::draw(void)
{
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT);
  renderer_->drawSprite(textureToShow_, glm::vec2{0.0f, 0.0f},
                        glm::vec2{
                            width_,
                            height_,
                        });
}

void GPUViewer::initTexture()
{
  textureToShow_.internalFormat = GL_RGBA;
  textureToShow_.imageFormat = GL_RGBA;
  textureToShow_.generate(width_, height_, nullptr);
  // textureToShow_.bind();
  HANDLE_ERROR(
      cudaGraphicsGLRegisterImage(&resource_, textureToShow_.ID, GL_TEXTURE_2D,
                                  cudaGraphicsRegisterFlagsWriteDiscard));
}
void GPUViewer::buildeShader()
{
  std::string vertSource{"./sprite.vert"};
  std::string fragSource{"./sprite.frag"};
  CHECK(fs::exist(vertSource))
  CHECK(fs::exist(fragSource))
  Shader shader =
      loadShaderFromFile(vertSource.c_str(), fragSource.c_str(), nullptr);
  {
    glm::mat4 projection = glm::ortho(0.0f, static_cast<float>(width_), 0.0f,
                                      static_cast<float>(height_), -1.0f, 1.0f);
    shader.use().setMatrix4("projection", projection);
  }
  renderer_ = std::make_unique<SpriteRenderer>(shader);
}