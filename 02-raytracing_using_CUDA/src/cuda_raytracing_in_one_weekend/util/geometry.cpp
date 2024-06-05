#include "geometry.h"

bool intersctionSegmentSegment(const glm::vec2 &a, const glm::vec2 &b,
                               const glm::vec2 &c, const glm::vec2 &d,
                               float &rayt) {
  auto ba = b - a;
  auto dc = d - c;
  if (glm::dot(glm::normalize(ba), glm::normalize(dc)) >= 0.999) {
    return false;
  }
  glm::mat2 m{ba[0], -dc[0], ba[1], -dc[1]};
  auto t = (c - a) * glm::inverse(m);
  if (t[0] >= 0.0f && t[0] <= 1.0f && t[1] >= 0 && t[1] <= 1.0f) {
    // intersection = a + ba * t[0];
    rayt = t[0];
    return true;
  }
  return false;
}
float rotateRadian(glm::vec2 &src, glm::vec2 &dst) {
  float cosValue = glm::dot(glm::normalize(src), glm::normalize(dst));
  float radian = glm::acos(cosValue);
  auto cross = glm::cross(glm::vec3{src, 0.0f}, glm::vec3{dst, 0.0f});
  if (cross[2] >= 0.0f) {
    return radian;
  } else {
    return -radian;
  }
}
