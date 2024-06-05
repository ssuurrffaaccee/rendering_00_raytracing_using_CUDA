#pragma once
#include "ecs/world.h"
#include "input/input.h"
#include "window/window.h"
struct Game {
  void init();
  void run();

 private:
  std::unique_ptr<World> world_{};
  std::unique_ptr<Window> window_{nullptr};
  std::unique_ptr<Input> input_{nullptr};
};
