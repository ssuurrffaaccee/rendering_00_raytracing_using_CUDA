#pragma once
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>


#include <array>
#include <functional>
#include <vector>

struct WindowCreateInfo {
  int width{1280};
  int height{1280};
  const char* title{"Ray Tracing in CUDA"};
  bool is_fullscreen{false};
};

class Window {
 public:
  Window() = default;
  ~Window();
  void initialize(WindowCreateInfo create_info);
  void pollEvents() const;
  bool shouldClose() const;
  void closeWindow() const;
  void setTitle(const char* title);
  GLFWwindow* getWindow() const;
  std::array<int, 2> getWindowSize() const;

  using onResetFunc = std::function<void()>;
  using onKeyFunc = std::function<void(int, int, int, int)>;
  using onCharFunc = std::function<void(unsigned int)>;
  using onCharModsFunc = std::function<void(int, unsigned int)>;
  using onMouseButtonFunc = std::function<void(int, int, int)>;
  using onCursorPosFunc = std::function<void(double, double)>;
  using onCursorEnterFunc = std::function<void(int)>;
  using onScrollFunc = std::function<void(double, double)>;
  using onDropFunc = std::function<void(int, const char**)>;
  using onWindowSizeFunc = std::function<void(int, int)>;
  using onWindowCloseFunc = std::function<void()>;

  void registerOnResetFunc(onResetFunc func) { onResetFunc_.push_back(func); }
  void registerOnKeyFunc(onKeyFunc func) { onKeyFunc_.push_back(func); }
  void registerOnCharFunc(onCharFunc func) { onCharFunc_.push_back(func); }
  void registerOnCharModsFunc(onCharModsFunc func) {
    onCharModsFunc_.push_back(func);
  }
  void registerOnMouseButtonFunc(onMouseButtonFunc func) {
    onMouseButtonFunc_.push_back(func);
  }
  void registerOnCursorPosFunc(onCursorPosFunc func) {
    onCursorPosFunc_.push_back(func);
  }
  void registerOnCursorEnterFunc(onCursorEnterFunc func) {
    onCursorEnterFunc_.push_back(func);
  }
  void registerOnScrollFunc(onScrollFunc func) {
    onScrollFunc_.push_back(func);
  }
  void registerOnDropFunc(onDropFunc func) { onDropFunc_.push_back(func); }
  void registerOnWindowSizeFunc(onWindowSizeFunc func) {
    onWindowSizeFunc_.push_back(func);
  }
  void registerOnWindowCloseFunc(onWindowCloseFunc func) {
    onWindowCloseFunc_.push_back(func);
  }

  bool isMouseButtonDown(int button) const {
    if (button < GLFW_MOUSE_BUTTON_1 || button > GLFW_MOUSE_BUTTON_LAST) {
      return false;
    }
    return glfwGetMouseButton(window_, button) == GLFW_PRESS;
  }
  bool getFocusMode() const { return is_focus_mode_; }
  void setFocusMode(bool mode);

 protected:
  // window event callbacks
  static void keyCallback(GLFWwindow* window, int key, int scancode, int action,
                          int mods) {
    Window* app = (Window*)glfwGetWindowUserPointer(window);
    if (app) {
      app->onKey(key, scancode, action, mods);
    }
  }
  static void charCallback(GLFWwindow* window, unsigned int codepoint) {
    Window* app = (Window*)glfwGetWindowUserPointer(window);
    if (app) {
      app->onChar(codepoint);
    }
  }
  static void charModsCallback(GLFWwindow* window, unsigned int codepoint,
                               int mods) {
    Window* app = (Window*)glfwGetWindowUserPointer(window);
    if (app) {
      app->onCharMods(codepoint, mods);
    }
  }
  static void mouseButtonCallback(GLFWwindow* window, int button, int action,
                                  int mods) {
    Window* app = (Window*)glfwGetWindowUserPointer(window);
    if (app) {
      app->onMouseButton(button, action, mods);
    }
  }
  static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    Window* app = (Window*)glfwGetWindowUserPointer(window);
    if (app) {
      app->onCursorPos(xpos, ypos);
    }
  }
  static void cursorEnterCallback(GLFWwindow* window, int entered) {
    Window* app = (Window*)glfwGetWindowUserPointer(window);
    if (app) {
      app->onCursorEnter(entered);
    }
  }
  static void scrollCallback(GLFWwindow* window, double xoffset,
                             double yoffset) {
    Window* app = (Window*)glfwGetWindowUserPointer(window);
    if (app) {
      app->onScroll(xoffset, yoffset);
    }
  }
  static void dropCallback(GLFWwindow* window, int count, const char** paths) {
    Window* app = (Window*)glfwGetWindowUserPointer(window);
    if (app) {
      app->onDrop(count, paths);
    }
  }
  static void windowSizeCallback(GLFWwindow* window, int width, int height) {
    Window* app = (Window*)glfwGetWindowUserPointer(window);
    if (app) {
      app->width_ = width;
      app->height_ = height;
    }
  }
  static void windowCloseCallback(GLFWwindow* window) {
    glfwSetWindowShouldClose(window, true);
  }

  void onReset() {
    for (auto& func : onResetFunc_) func();
  }
  void onKey(int key, int scancode, int action, int mods) {
    for (auto& func : onKeyFunc_) func(key, scancode, action, mods);
  }
  void onChar(unsigned int codepoint) {
    for (auto& func : onCharFunc_) func(codepoint);
  }
  void onCharMods(int codepoint, unsigned int mods) {
    for (auto& func : onCharModsFunc_) func(codepoint, mods);
  }
  void onMouseButton(int button, int action, int mods) {
    for (auto& func : onMouseButtonFunc_) func(button, action, mods);
  }
  void onCursorPos(double xpos, double ypos) {
    for (auto& func : onCursorPosFunc_) func(xpos, ypos);
  }
  void onCursorEnter(int entered) {
    for (auto& func : onCursorEnterFunc_) func(entered);
  }
  void onScroll(double xoffset, double yoffset) {
    for (auto& func : onScrollFunc_) func(xoffset, yoffset);
  }
  void onDrop(int count, const char** paths) {
    for (auto& func : onDropFunc_) func(count, paths);
  }
  void onWindowSize(int width, int height) {
    for (auto& func : onWindowSizeFunc_) func(width, height);
  }

 private:
  GLFWwindow* window_{nullptr};
  int width_{0};
  int height_{0};

  bool is_focus_mode_{false};

  std::vector<onResetFunc> onResetFunc_;
  std::vector<onKeyFunc> onKeyFunc_;
  std::vector<onCharFunc> onCharFunc_;
  std::vector<onCharModsFunc> onCharModsFunc_;
  std::vector<onMouseButtonFunc> onMouseButtonFunc_;
  std::vector<onCursorPosFunc> onCursorPosFunc_;
  std::vector<onCursorEnterFunc> onCursorEnterFunc_;
  std::vector<onScrollFunc> onScrollFunc_;
  std::vector<onDropFunc> onDropFunc_;
  std::vector<onWindowSizeFunc> onWindowSizeFunc_;
  std::vector<onWindowCloseFunc> onWindowCloseFunc_;
};