#include "input.h"

#include <array>

#include "cursorPos.h"
#include "ecs/world.h"
#include "inputStatus.h"
#include "util/check.h"
#include "window/window.h"

void Input::onKey(int key, int scancode, int action, int mods) {
  if (action == GLFW_PRESS) {
    switch (key) {
      case GLFW_KEY_ESCAPE:
        window_->closeWindow();
        break;
      case GLFW_KEY_A:
        m_game_command |= (unsigned int)GameCommand::left;
        break;
      case GLFW_KEY_S:
        m_game_command |= (unsigned int)GameCommand::backward;
        break;
      case GLFW_KEY_W:
        m_game_command |= (unsigned int)GameCommand::forward;
        break;
      case GLFW_KEY_D:
        m_game_command |= (unsigned int)GameCommand::right;
        break;
      case GLFW_KEY_SPACE:
        window_->setFocusMode(!window_->getFocusMode());
        break;
      default:
        break;
    }
  }
  if (action == GLFW_REPEAT) {
    switch (key) {
      case GLFW_KEY_A:
        m_game_command |= (unsigned int)GameCommand::left;
        break;
      case GLFW_KEY_S:
        m_game_command |= (unsigned int)GameCommand::backward;
        break;
      case GLFW_KEY_W:
        m_game_command |= (unsigned int)GameCommand::forward;
        break;
      case GLFW_KEY_D:
        m_game_command |= (unsigned int)GameCommand::right;
        break;
      default:
        break;
    }
  }
}

void Input::onCursorPos(double current_cursor_x, double current_cursor_y) {
  // if (window_->getFocusMode()) {
  cursor_delta_x_ = current_cursor_x - last_cursor_x_;
  cursor_delta_y_ = current_cursor_y - last_cursor_y_;
  // }
  last_cursor_x_ = current_cursor_x;
  last_cursor_y_ = current_cursor_y;
  auto* cursorPos = world_->getResource<CursorPos>();
  cursorPos->position_ =
      glm::vec2{float(last_cursor_x_), float(last_cursor_y_)};
  cursorPos->positionOffset_ = glm::vec2{cursor_delta_x_, cursor_delta_y_};
}
void Input::onScroll(double xoffset, double yoffset) {
  auto* cursorPos = world_->getResource<CursorPos>();
  cursorPos->scroll_ = glm::vec2{xoffset, yoffset};
}

void Input::clear() {
  cursor_delta_x_ = 0;
  cursor_delta_y_ = 0;
}

void Input::initialize(Window* window, World* world) {
  window_ = window;
  CHECK(window_ != nullptr);
  world_ = world;
  CHECK(world != nullptr);
  world_->addResource(InputStatus{uint32_t(0)});
  world->addResource(CursorPos{glm::vec2{0.0f, 0.0f}});
  window_->registerOnKeyFunc(
      [this](int key, int scancode, int action, int mods) {
        this->onKey(key, scancode, action, mods);
      });
  window_->registerOnCursorPosFunc(
      [this](double current_cursor_x, double current_cursor_y) {
        this->onCursorPos(current_cursor_x, current_cursor_y);
      });
  window_->registerOnScrollFunc([this](double xoffset, double yoffset) {
    this->onScroll(xoffset, yoffset);
  });
}

void Input::update() {
  clear();
  InputStatus* inputStatus = world_->getResource<InputStatus>();
  if (window_->getFocusMode()) {
    m_game_command |= (unsigned int)GameCommand::focus;
  }
  inputStatus->gameCommand_ = m_game_command;
  m_game_command = uint32_t(0);
}
