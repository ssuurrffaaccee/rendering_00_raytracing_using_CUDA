#pragma once
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "component.h"
#include "system.h"
struct World {
  template <class T>
  T* getComponent(EntityID entityID) {
    auto componentTypeID = ComponentTypeIDGetter::get<T>();
    auto componentIter = ComponentEntityMap_.find(componentTypeID);
    if (componentIter == ComponentEntityMap_.end()) {
      return nullptr;
    }
    auto entityIter = componentIter->second.find(entityID);
    if (entityIter == componentIter->second.end()) {
      return nullptr;
    }
    return (T*)entityIter->second.data_;
  }
  template <class T>
  void addComponent(EntityID entityID, const T& componentData) {
    auto componentTypeID = ComponentTypeIDGetter::get<T>();
    auto& component = ComponentEntityMap_[componentTypeID][entityID];
    component.data_ = (void*)new T(componentData);
    component.destroyFunc_ = [](void* data) { delete (T*)data; };
    componentTypeIDToEntities_[componentTypeID].insert(entityID);
  }
  template <class T>
  void addResource(const T& componentData) {
    auto componentTypeID = ComponentTypeIDGetter::get<T>();
    auto& component = resources_[componentTypeID];
    component.data_ = (void*)new T(componentData);
    component.destroyFunc_ = [](void* data) { delete (T*)data; };
  }
  template <class T>
  T* getResource() {
    auto componentTypeID = ComponentTypeIDGetter::get<T>();
    auto iter = resources_.find(componentTypeID);
    if (iter == resources_.end()) {
      return nullptr;
    }
    return (T*)iter->second.data_;
  }
  template <class T>
  std::unordered_set<EntityID>& getEntities() {
    auto componentTypeID = ComponentTypeIDGetter::get<T>();
    auto iter = componentTypeIDToEntities_.find(componentTypeID);
    if (iter == componentTypeIDToEntities_.end()) {
      return componentTypeIDToEntities_[componentTypeID];
    }
    return iter->second;
  }
  template <class T>
  void addUpdateSystem(const std::shared_ptr<T>& system) {
    updateSystems_.push_back(std::dynamic_pointer_cast<System>(system));
  }
  template <class T>
  void addStartSystem(const std::shared_ptr<T>& system) {
    startSystems_.push_back(std::dynamic_pointer_cast<System>(system));
  }
  void start() {
    for (auto& system : startSystems_) {
      system->start(*this);
    }
    for (auto& system : updateSystems_) {
      system->start(*this);
    }
  }
  void update(float dt) {
    for (auto& system : updateSystems_) {
      system->update(dt, *this);
    }
  }
  void render() {
    for (auto& system : updateSystems_) {
      system->render(*this);
    }
  }

 private:
  std::unordered_map<ComponentTypeID, std::unordered_map<EntityID, Component>>
      ComponentEntityMap_;
  std::unordered_map<ComponentTypeID, std::unordered_set<EntityID>>
      componentTypeIDToEntities_;
  std::unordered_map<ComponentTypeID, Component> resources_;
  std::vector<std::shared_ptr<System>> updateSystems_;
  std::vector<std::shared_ptr<System>> startSystems_;
};

inline std::unordered_set<EntityID>& min(std::unordered_set<EntityID>& a,
                                         std::unordered_set<EntityID>& b) {
  if (a.size() < b.size()) {
    return a;
  } else {
    return b;
  }
}
inline std::unordered_set<EntityID>& min(std::unordered_set<EntityID>& a,
                                         std::unordered_set<EntityID>& b,std::unordered_set<EntityID>& c) {
  if(a.size() < b.size() && a.size()<c.size()){
    return a;
  }
  if(b.size() < a.size() && b.size()<c.size()){
    return b;
  }
  return c;
}