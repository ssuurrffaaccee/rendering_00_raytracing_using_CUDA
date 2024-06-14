#include "common.h"
struct RotateMe {
  glm::vec3 origin{0.0f, 0.0f, 0.0f};
  glm::vec3 axis{0.0, 1.0f, 0.0f};
  float speed{1.0f};
  bool aroundSelf{false};
};

struct HeadTo {
  glm::vec3 target{0.0f, 0.0f, 0.0f};
};