#include "shader.hpp"
#include <glad/glad.h>
#include <iostream>
#include <fstream>
#include <sstream>

Shader &Shader::use() {
  glUseProgram(this->ID);
  return *this;
}

void Shader::compile(const char *vertexSource, const char *fragmentSource,
                     const char *geometrySource) {
  unsigned int sVertex, sFragment, gShader;
  // vertex Shader
  sVertex = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(sVertex, 1, &vertexSource, NULL);
  glCompileShader(sVertex);
  checkCompileErrors(sVertex, "VERTEX");
  // fragment Shader
  sFragment = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(sFragment, 1, &fragmentSource, NULL);
  glCompileShader(sFragment);
  checkCompileErrors(sFragment, "FRAGMENT");
  // if geometry shader source code is given, also compile geometry shader
  if (geometrySource != nullptr) {
    gShader = glCreateShader(GL_GEOMETRY_SHADER);
    glShaderSource(gShader, 1, &geometrySource, NULL);
    glCompileShader(gShader);
    checkCompileErrors(gShader, "GEOMETRY");
  }
  // shader program
  this->ID = glCreateProgram();
  glAttachShader(this->ID, sVertex);
  glAttachShader(this->ID, sFragment);
  if (geometrySource != nullptr) glAttachShader(this->ID, gShader);
  glLinkProgram(this->ID);
  checkCompileErrors(this->ID, "PROGRAM");
  // delete the shaders as they're linked into our program now and no longer
  // necessary
  glDeleteShader(sVertex);
  glDeleteShader(sFragment);
  if (geometrySource != nullptr) glDeleteShader(gShader);
}

void Shader::setFloat(const std::string& name, float value, bool useShader) {
  if (useShader) this->use();
  glUniform1f(glGetUniformLocation(this->ID, name.c_str()), value);
}
void Shader::setInteger(const std::string& name, int value, bool useShader) {
  if (useShader) this->use();
  glUniform1i(glGetUniformLocation(this->ID, name.c_str()), value);
}
void Shader::setVector2f(const std::string& name, float x, float y, bool useShader) {
  if (useShader) this->use();
  glUniform2f(glGetUniformLocation(this->ID, name.c_str()), x, y);
}
void Shader::setVector2f(const std::string& name, const glm::vec2 &value,
                         bool useShader) {
  if (useShader) this->use();
  glUniform2f(glGetUniformLocation(this->ID, name.c_str()), value.x, value.y);
}
void Shader::setVector3f(const std::string& name, float x, float y, float z,
                         bool useShader) {
  if (useShader) this->use();
  glUniform3f(glGetUniformLocation(this->ID, name.c_str()), x, y, z);
}
void Shader::setVector3f(const std::string& name, const glm::vec3 &value,
                         bool useShader) {
  if (useShader) this->use();
  glUniform3f(glGetUniformLocation(this->ID, name.c_str()), value.x, value.y, value.z);
}
void Shader::setVector4f(const std::string& name, float x, float y, float z, float w,
                         bool useShader) {
  if (useShader) this->use();
  glUniform4f(glGetUniformLocation(this->ID, name.c_str()), x, y, z, w);
}
void Shader::setVector4f(const std::string& name, const glm::vec4 &value,
                         bool useShader) {
  if (useShader) this->use();
  glUniform4f(glGetUniformLocation(this->ID, name.c_str()), value.x, value.y, value.z,
              value.w);
}
void Shader::setMatrix4(const std::string& name, const glm::mat4 &matrix,
                        bool useShader) {
  if (useShader) this->use();
  glUniformMatrix4fv(glGetUniformLocation(this->ID, name.c_str()), 1, false,
                     glm::value_ptr(matrix));
}

void Shader::checkCompileErrors(unsigned int object, std::string type) {
  int success;
  char infoLog[1024];
  if (type != "PROGRAM") {
    glGetShaderiv(object, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(object, 1024, NULL, infoLog);
      std::cout
          << "| ERROR::SHADER: Compile-time error: Type: " << type << "\n"
          << infoLog
          << "\n -- --------------------------------------------------- -- "
          << std::endl;
    }
  } else {
    glGetProgramiv(object, GL_LINK_STATUS, &success);
    if (!success) {
      glGetProgramInfoLog(object, 1024, NULL, infoLog);
      std::cout
          << "| ERROR::Shader: Link-time error: Type: " << type << "\n"
          << infoLog
          << "\n -- --------------------------------------------------- -- "
          << std::endl;
    }
  }
}

Shader loadShaderFromFile(const char *vShaderFile,
                                           const char *fShaderFile,
                                           const char *gShaderFile) {
  // 1. retrieve the vertex/fragment source code from filePath
  std::string vertexCode;
  std::string fragmentCode;
  std::string geometryCode;
  try {
    // open files
    std::ifstream vertexShaderFile(vShaderFile);
    std::ifstream fragmentShaderFile(fShaderFile);
    std::stringstream vShaderStream, fShaderStream;
    // read file's buffer contents into streams
    vShaderStream << vertexShaderFile.rdbuf();
    fShaderStream << fragmentShaderFile.rdbuf();
    // close file handlers
    vertexShaderFile.close();
    fragmentShaderFile.close();
    // convert stream into string
    vertexCode = vShaderStream.str();
    fragmentCode = fShaderStream.str();
    std::cout << "vert: " << vertexCode << "\n\n frag: " << fragmentCode
              << "\n";
    // if geometry shader path is present, also load a geometry shader
    if (gShaderFile != nullptr) {
      std::ifstream geometryShaderFile(gShaderFile);
      std::stringstream gShaderStream;
      gShaderStream << geometryShaderFile.rdbuf();
      geometryShaderFile.close();
      geometryCode = gShaderStream.str();
    }
  } catch (std::exception e) {
    std::cout << "ERROR::SHADER: Failed to read shader files" << std::endl;
  }
  const char *vShaderCode = vertexCode.c_str();
  const char *fShaderCode = fragmentCode.c_str();
  const char *gShaderCode = geometryCode.c_str();
  // 2. now create shader object from source code
  Shader shader;
  shader.compile(vShaderCode, fShaderCode,
                 gShaderFile != nullptr ? gShaderCode : nullptr);
  return shader;
}