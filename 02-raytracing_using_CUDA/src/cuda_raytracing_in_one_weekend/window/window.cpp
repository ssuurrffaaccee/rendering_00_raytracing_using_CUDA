#include "window.h"

#include "util/check.h"

static void glfwErrorPrint(int code, const char* info) {
  std::cout << "GLFW error! code: " << code << " info: " << info << "\n";
}

// int width = 0, height = 0;
// glfwGetFramebufferSize(window, &width, &height);
// while (width == 0 || height == 0) {
//     glfwGetFramebufferSize(window, &width, &height);
//     glfwWaitEvents();
// }

Window::~Window() {
  glfwDestroyWindow(window_);
  glfwTerminate();
}

void Window::initialize(WindowCreateInfo create_info) {
  glfwSetErrorCallback(&glfwErrorPrint);
  CHECK_WITH_INFO(glfwInit() == GLFW_TRUE, "Failed to initialize GLFW");
  width_ = create_info.width;
  height_ = create_info.height;
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_SCALE_FRAMEBUFFER, GLFW_FALSE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
  glfwWindowHint(GLFW_RESIZABLE, false);
  // glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  window_ = glfwCreateWindow(create_info.width, create_info.height,
                             create_info.title, nullptr, nullptr);
  CHECK_WITH_INFO_THEN(window_ != nullptr, "Failed to create window",
                       glfwTerminate();)

  // glad: load all OpenGL function pointers
  // ---------------------------------------
  glfwMakeContextCurrent(window_);
  CHECK_WITH_INFO(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress),
                  "Failed to initialize GLAD");
  // Setup input callbacks
  glfwSetWindowUserPointer(window_, this);
  glfwSetKeyCallback(window_, keyCallback);
  glfwSetCharCallback(window_, charCallback);
  glfwSetCharModsCallback(window_, charModsCallback);
  glfwSetMouseButtonCallback(window_, mouseButtonCallback);
  glfwSetCursorPosCallback(window_, cursorPosCallback);
  glfwSetCursorEnterCallback(window_, cursorEnterCallback);
  glfwSetScrollCallback(window_, scrollCallback);
  glfwSetDropCallback(window_, dropCallback);
  glfwSetWindowSizeCallback(window_, windowSizeCallback);
  glfwSetWindowCloseCallback(window_, windowCloseCallback);

  glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

void Window::pollEvents() const { glfwPollEvents(); }

bool Window::shouldClose() const { return glfwWindowShouldClose(window_); }
void Window::closeWindow() const {
  glfwSetWindowShouldClose(window_, GLFW_TRUE);
}

void Window::setTitle(const char* title) { glfwSetWindowTitle(window_, title); }

GLFWwindow* Window::getWindow() const { return window_; }

std::array<int, 2> Window::getWindowSize() const {
  return std::array<int, 2>({width_, height_});
}

void Window::setFocusMode(bool mode) {
  is_focus_mode_ = mode;
  glfwSetInputMode(window_, GLFW_CURSOR,
                   is_focus_mode_ ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
}