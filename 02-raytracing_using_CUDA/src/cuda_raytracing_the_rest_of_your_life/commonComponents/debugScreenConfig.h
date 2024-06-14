#pragma once 
#include "common.h"
#include <vector>
struct DebugScreenConfig{
   struct DebugScreen{
      glm::vec2 position;
      glm::vec2 size;
   };
   std::vector<DebugScreen> screens;
};