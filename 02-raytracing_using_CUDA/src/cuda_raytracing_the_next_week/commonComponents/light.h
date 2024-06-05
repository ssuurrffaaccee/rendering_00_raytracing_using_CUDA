#pragma once
#include <glm/glm.hpp>
inline const static int MAX_POINT_LGITH_NUM{32};
struct PointLight {
  float linear_{0.7f};
  float quadratic_{1.8f};
};
// enum class LightType {
//   UNKOWN,
//   DIRECTION,
//   POINT,
//   SPOT,
// };
// struct Light {
//   LightType type_{LightType::UNKOWN};
//   float cutOff_;
//   float outerCutOff_;

//   float constant_;
//   float linear_;
//   float quadratic_;

//   glm::vec3 ambient_;
//   glm::vec3 diffuse_;
//   glm::vec3 specular_;
// };
struct LightMaterial {
  glm::vec3 ambient_;
  glm::vec3 diffuse_;
  glm::vec3 specular_;
};
struct LightAttenuation {
  float constant_;
  float linear_;
  float quadratic_;
};
struct DirLight {
  glm::vec3 direction_;
  LightMaterial material_;
};
struct _PointLight {
  LightMaterial material_;
  LightAttenuation attenuation_;
};
struct SpotLight {
  glm::vec3 direction;
  float cutOff;
  float outerCutOff;
  LightAttenuation attenuation_;
  LightMaterial material_;
};

struct MaterialConfig{
    float shininess;
};