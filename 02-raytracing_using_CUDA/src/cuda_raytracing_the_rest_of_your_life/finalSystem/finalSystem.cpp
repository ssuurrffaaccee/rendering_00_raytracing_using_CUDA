#include "finalSystem.h"

#include "ecs/system.h"
#include "ecs/world.h"
#include "input/cursorPos.h"
void FinalSystem::start(World& world) {}
void FinalSystem::update(float dt, World& world) {
  auto* cursorPos = world.getResource<CursorPos>();
  cursorPos->scroll_[0] = 0.0f;
  cursorPos->scroll_[1] = 0.0f;
  cursorPos->positionOffset_[0] = 0.0f;
  cursorPos->positionOffset_[1] = 0.0f;
}
void FinalSystem::render(World& world) {}