#include "color.h"

#include "util/check.h"
#include "util/random.h"
glm::vec3 BaseColor::getColor() {
  if (baseColor_.has_value()) {
    return baseColor_.value();
  } else {
    if (type_ == BaseColorType::FIXED) {
      baseColor_ = glm::vec3{1.0f, 1.0f, 1.0f};
    } else if (type_ == BaseColorType::RNADOM) {
      baseColor_ = glm::vec3{myRandom(), myRandom(), myRandom()};
    } else {
      CHECK_WITH_INFO(false, "unkown base color type")
      baseColor_ = glm::vec3{};
    }
    return baseColor_.value();
  }
}