#pragma once
class World;
struct System {
  virtual ~System() {}
  virtual void start(World& world) {}
  virtual void update(float dt, World& world) {}
  virtual void render(World& world) {}
};
