

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "cameraSystem/camera.h"
#include "window/windowSize.h"
#include "util/fs.hpp"
#include "resources/resourceManager.h"
#include "CUDARayTracingSystem.h"
CUDARayTracingSystem::CUDARayTracingSystem() {}

CUDARayTracingSystem::~CUDARayTracingSystem() {
  glDeleteTextures(1, &textureToShow_.ID);
}
void CUDARayTracingSystem::start(World &world) {
  {
    std::string vertSource{"./sprite.vert"};
    std::string fragSource{"./sprite.frag"};
    CHECK(fs::exist(vertSource))
    CHECK(fs::exist(fragSource))
    ResourceManager::loadShader(vertSource.c_str(), fragSource.c_str(), nullptr,
                                "sprit_renderer");
  }

  auto windowSize = world.getResource<WindowSize>();
  CHECK(windowSize != nullptr);
  textureToShow_.internalFormat=GL_RGBA;
  textureToShow_.imageFormat=GL_RGBA;
  textureToShow_.generate(windowSize->width_, windowSize->height_, nullptr);
  rayTracingDriver_.init(this->textureToShow_, windowSize->width_,
                         windowSize->height_);
  {
    {
      glm::mat4 projection =
          glm::ortho(0.0f, static_cast<float>(windowSize->width_), 0.0f,
                     static_cast<float>(windowSize->height_), -1.0f, 1.0f);
      renderer_ = std::make_unique<SpriteRenderer>(
          ResourceManager::getShader("sprit_renderer"));
      ResourceManager::getShader("sprit_renderer")
          .use()
          .setMatrix4("projection", projection);
    }
  }
}
void CUDARayTracingSystem::render(World &world) {
  auto *camera = world.getResource<Camera>();
  CHECK(camera != nullptr);
  auto *windowSize = world.getResource<WindowSize>();
  CHECK(windowSize != nullptr);
  rayTracingDriver_.render(*camera);
  // {// debug
  // int w = 100;
  // int h = 100;
  // std::vector<float> data;
  // data.resize(w * h * 4, 0.5);
  // glBindTexture(GL_TEXTURE_2D, textureToShow_.ID);
  // glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 100, 100,
  //                 GL_RGBA, GL_FLOAT, data.data());
  // }

  renderer_->drawSprite(textureToShow_, glm::vec2{0.0f, 0.0f},
                        glm::vec2{
                            windowSize->width_,
                            windowSize->height_,
                        });
}