#pragma once
#include "ecs/system.h"
#include "ecs/world.h"
class FinalSystem : public System {
 public:
  FinalSystem() {}
  virtual ~FinalSystem(){};
  void start(World& world) override;
  void update(float dt, World& world) override;
  void render(World& world) override;
};