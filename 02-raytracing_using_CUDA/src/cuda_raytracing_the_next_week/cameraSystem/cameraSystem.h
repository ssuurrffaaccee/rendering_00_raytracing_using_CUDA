#pragma once
#include "camera.h"
#include "ecs/system.h"
#include "ecs/world.h"
#include "util/check.h"

struct CameraSystem : public System {
  CameraSystem() {}
  virtual ~CameraSystem() {}
  void start(World& world) override;
  void update(float dt, World& world) override;
};