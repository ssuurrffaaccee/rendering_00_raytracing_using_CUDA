#pragma once
#include <functional>

#include "id.h"
struct Component {
  void* data_{nullptr};
  using DestroyFunc = std::function<void(void*)>;
  DestroyFunc destroyFunc_;
  ~Component() {
    if (destroyFunc_) {
      destroyFunc_(data_);
    }
  }
};