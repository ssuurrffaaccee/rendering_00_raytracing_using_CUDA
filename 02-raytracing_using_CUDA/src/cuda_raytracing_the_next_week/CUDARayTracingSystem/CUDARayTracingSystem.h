#pragma once

#include "ecs/system.h"
#include "ecs/world.h"
#include "resources/texture.h"
#include "raytracingDriver.h"
#include "util/check.h"
#include "renderers/spriteRenderer.h"

struct CUDARayTracingSystem : public System {
  CUDARayTracingSystem();
  virtual ~CUDARayTracingSystem();
  void start(World& world) override;
  void update(float dt, World& world) override{}
  void render(World &world) override;
 private:
  Texture2D textureToShow_{};
  RayTracingDriver rayTracingDriver_{};
  std::unique_ptr<SpriteRenderer> renderer_{nullptr};
};