#include "viewerCPU.hpp"
static void glfwErrorPrint(int code, const char *info)
{
    std::cout << "GLFW error! code: " << code << " info: " << info << "\n";
}
void CPUViewer::init(int width, int height)
{
    width_ = width;
    height_ = height;
    pixels_.resize(width_ * height_ * 4, 100);
    std::cout << pixels_.size() << "\n";
}
void CPUViewer::displayAndExit(std::function<void(unsigned char *, int, int)> updateFunc)
{
    initWindowAndOpengGL();
    registeCallback();
    buildeShader();
    initTexture();
    if (updateFunc == nullptr)
    {
        copyToTexture();
        while (!glfwWindowShouldClose(window_))
        {
            glfwPollEvents();
            draw();
            glfwSwapBuffers(window_);
        }
    }
    else
    {
        while (!glfwWindowShouldClose(window_))
        {
            glfwPollEvents();
            updateFunc(this->pixels_.data(), width_, height_);
            copyToTexture();
            draw();
            glfwSwapBuffers(window_);
        }
    }
}
CPUViewer::~CPUViewer()
{
    glDeleteTextures(1, &textureToShow_.ID);
    glfwDestroyWindow(window_);
    glfwTerminate();
}
void CPUViewer::initWindowAndOpengGL()
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
void CPUViewer::registeCallback()
{
    glfwSetWindowUserPointer(window_, this);
    glfwSetKeyCallback(window_, keyCallback);
}
void CPUViewer::keyCallback(GLFWwindow *window, int key, int scancode, int action,
                            int mods)
{
    // CPUViewer* app = (CPUViewer*)glfwGetWindowUserPointer(window);
    if (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}
void CPUViewer::draw(void)
{
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);
    renderer_->drawSprite(textureToShow_, glm::vec2{0.0f, 0.0f},
                          glm::vec2{
                              width_,
                              height_,
                          });
}

void CPUViewer::initTexture()
{
    textureToShow_.internalFormat = GL_RGBA;
    textureToShow_.imageFormat = GL_RGBA;
    textureToShow_.generate(width_, height_, nullptr);
}
void CPUViewer::copyToTexture()
{
    unsigned char *dataPtr = this->pixels_.data();
    glBindTexture(GL_TEXTURE_2D, textureToShow_.ID);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA,
                    GL_UNSIGNED_BYTE, dataPtr);
    // { // debug
    //     int w = 100;
    //     int h = 100;
    //     std::vector<float> data;
    //     data.resize(w * h * 4, 0.5);
    //     glBindTexture(GL_TEXTURE_2D, textureToShow_.ID);
    //     glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 100, 100,
    //                     GL_RGBA, GL_FLOAT, data.data());
    // }
}
void CPUViewer::buildeShader()
{
    std::string vertSource{"./sprite.vert"};
    std::string fragSource{"./sprite.frag"};
    CHECK(fs::exist(vertSource))
    CHECK(fs::exist(fragSource))
    Shader shader =
        loadShaderFromFile(vertSource.c_str(), fragSource.c_str(), nullptr);
    {
        glm::mat4 projection =
            glm::ortho(0.0f, static_cast<float>(width_), 0.0f,
                       static_cast<float>(height_), -1.0f, 1.0f);
        shader.use().setMatrix4("projection", projection);
    }
    renderer_ = std::make_unique<SpriteRenderer>(shader);
}