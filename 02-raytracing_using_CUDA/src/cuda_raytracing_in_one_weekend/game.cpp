#include "game.h"

#include "cameraSystem/cameraSystem.h"
#include "ecs/world.h"
#include "entityInitSystem/entityInitSystem.h"
#include "finalSystem/finalSystem.h"
#include "input/input.h"
#include "window/window.h"
#include "window/windowSize.h"
#include "CUDARayTracingSystem/CUDARayTracingSystem.h"

void Game::init() {
  world_ = std::make_unique<World>();
  window_ = std::make_unique<Window>();
  input_ = std::make_unique<Input>();
  WindowCreateInfo windowCreateInfo;
  window_->initialize(windowCreateInfo);
  input_->initialize(window_.get(), world_.get());
  /////////////////////////////////////////////////////////////////
  auto [width, height] = window_->getWindowSize();
  world_->addResource(WindowSize{width, height});
  world_->addStartSystem(std::make_shared<EntityInitSystem>());
  world_->addUpdateSystem(std::make_shared<CameraSystem>());
  world_->addUpdateSystem(std::make_shared<CUDARayTracingSystem>());
  ////////////////////////////////////////////////////////////////
  world_->addUpdateSystem(std::make_shared<FinalSystem>());
  glViewport(0, 0, width, height);
}

void Game::run() {
  world_->start();
  // glEnable(GL_BLEND);
  // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  // glFrontFace(GL_CCW);
  float deltaTime = 0.0f;
  float lastFrame = 0.0f;
  while (!window_->shouldClose()) {
    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;
    window_->pollEvents();
    input_->update();
    world_->update(deltaTime);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    world_->render();
    glfwSwapBuffers(window_->getWindow());
  }
}