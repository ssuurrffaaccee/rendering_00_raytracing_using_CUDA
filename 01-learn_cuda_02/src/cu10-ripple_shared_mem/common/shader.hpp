#ifndef SHADER_H
#define SHADER_H
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>
class Shader {
 public:
  unsigned int ID;
  Shader() {}
  Shader &use();
  void compile(const char *vertexSource, const char *fragmentSource,
               const char *geometrySource =
                   nullptr);  // note: geometry source code is optional
  void setFloat(const std::string &name, float value, bool useShader = false);
  void setInteger(const std::string &name, int value, bool useShader = false);
  void setVector2f(const std::string &name, float x, float y,
                   bool useShader = false);
  void setVector2f(const std::string &name, const glm::vec2 &value,
                   bool useShader = false);
  void setVector3f(const std::string &name, float x, float y, float z,
                   bool useShader = false);
  void setVector3f(const std::string &name, const glm::vec3 &value,
                   bool useShader = false);
  void setVector4f(const std::string &name, float x, float y, float z, float w,
                   bool useShader = false);
  void setVector4f(const std::string &name, const glm::vec4 &value,
                   bool useShader = false);
  void setMatrix4(const std::string &name, const glm::mat4 &matrix,
                  bool useShader = false);

 private:
  // checks if compilation or linking failed and if so, print the error logs
  void checkCompileErrors(unsigned int object, std::string type);
};

Shader loadShaderFromFile(const char *vShaderFile, const char *fShaderFile,
                          const char *gShaderFile);
#endif