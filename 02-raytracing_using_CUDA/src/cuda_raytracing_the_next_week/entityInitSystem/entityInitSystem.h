#pragma once
#include "ecs/system.h"
#include "ecs/world.h"
struct EntityInitSystem : public System {
  EntityInitSystem() {}
  virtual ~EntityInitSystem() {}
  void start(World& world) override;
};