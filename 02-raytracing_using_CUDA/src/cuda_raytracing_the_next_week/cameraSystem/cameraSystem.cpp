#include "cameraSystem.h"

#include <bitset>

#include "input/cursorPos.h"
#include "input/input.h"
#include "input/inputStatus.h"
#include "util/geometry.h"
#include "util/toString.h"
#include "window/windowSize.h"
namespace {
void processKey(float dt, Camera& camera, InputStatus& inputStatus) {
  if (inputStatus.gameCommand_ == 0) {
    return;
  }
  if (inputStatus.gameCommand_ & (uint32_t)GameCommand::forward) {
    // std::cout << "forward\n";
    camera.processKeyboard(Camera_Movement::FORWARD, dt);
  }
  if (inputStatus.gameCommand_ & (uint32_t)GameCommand::backward) {
    // std::cout << "backward\n";
    camera.processKeyboard(Camera_Movement::BACKWARD, dt);
  }
  if (inputStatus.gameCommand_ & (uint32_t)GameCommand::left) {
    // std::cout << "left\n";
    camera.processKeyboard(Camera_Movement::LEFT, dt);
  }
  if (inputStatus.gameCommand_ & (uint32_t)GameCommand::right) {
    // std::cout << "right\n";
    camera.processKeyboard(Camera_Movement::RIGHT, dt);
  }
}
void processMouse(Camera& camera, CursorPos& cursorPos) {
  float offsetX = clamp(cursorPos.positionOffset_[0], -5.0f, 5.0f);
  float offsetY = clamp(cursorPos.positionOffset_[1], -5.0f, 5.0f);
  // float offsetX = cursorPos.positionOffset_[0];
  // float offsetY = cursorPos.positionOffset_[1];
  // std::cout << " offset: " << toString(cursorPos.positionOffset_);
  camera.processMouseMovement(offsetX, -offsetY);
  float scroll = clamp(cursorPos.scroll_[1], -3.0f, 3.0f);
  // std::cout << "scroll: " << toString(cursorPos.scroll_);
  camera.processMouseScroll(scroll);
}
}  // namespace
void CameraSystem::start(World& world) {
  world.addResource<Camera>(Camera{glm::vec3{478, 278, -600}});
}
void CameraSystem::update(float dt, World& world) {
  auto* camera = world.getResource<Camera>();
  auto* cursorPos = world.getResource<CursorPos>();
  auto* inputStatus = world.getResource<InputStatus>();
  if (inputStatus->gameCommand_ & (uint32_t)GameCommand::focus) {
    processKey(dt, *camera, *inputStatus);
    processMouse(*camera, *cursorPos);
  }
}