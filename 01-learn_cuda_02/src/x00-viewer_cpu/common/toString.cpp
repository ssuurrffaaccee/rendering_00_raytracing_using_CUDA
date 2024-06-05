#include "toString.hpp"
std::string toString(const glm::mat4& m4) {
  std::stringstream ss;
  ss << "mat4[\n";
  ss << m4[0][0] << "," << m4[1][0] << "," << m4[2][0] << "," << m4[3][0]
     << "\n";
  ss << m4[0][1] << "," << m4[1][1] << "," << m4[2][1] << "," << m4[3][1]
     << "\n";
  ss << m4[0][2] << "," << m4[1][2] << "," << m4[2][2] << "," << m4[3][2]
     << "\n";
  ss << m4[0][3] << "," << m4[1][3] << "," << m4[2][3] << "," << m4[3][3]
     << "]\n";
  return ss.str();
}
std::string toString(const glm::mat2& m2) {
  std::stringstream ss;
  ss << "mat2[\n";
  ss << m2[0][0] << "," << m2[1][0] << "\n";
  ss << m2[0][1] << "," << m2[1][1] << "\n";
  return ss.str();
}
std::string toString(const glm::vec2& v2) {
  std::stringstream ss;
  ss << "vec2(" << v2[0] << "," << v2[1] << ")\n";
  return ss.str();
}
std::string toString(const glm::vec3& v3) {
  std::stringstream ss;
  ss << "vec3(" << v3[0] << "," << v3[1] << "," << v3[2] << ")\n";
  return ss.str();
}
std::string toString(const glm::vec4& v4){
   std::stringstream ss;
  ss << "vec4(" << v4[0] << "," << v4[1] << "," << v4[2] << "," << v4[3] << ")\n";
  return ss.str(); 
}