#pragma once
#include <cstdint>
using EntityID = uint32_t;
using ComponentTypeID = uint32_t;

struct ComponentTypeIDGetter {
  template <class T>
  static ComponentTypeID get() {
    static ComponentTypeID id = nextId_++;
    return id;
  }
  static uint32_t nextId_;
};


struct EntityIDGenerator {
  static EntityID alloc() {
    static uint32_t nextId_{0};
    return nextId_++;
  }
};
