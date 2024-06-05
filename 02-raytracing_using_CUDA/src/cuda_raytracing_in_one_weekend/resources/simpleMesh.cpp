#include "simpleMesh.h"

#include <glad/glad.h>
SimpleMesh loadCube() {
  // postion:vec3, normal:vec3, texcoords:vec2
  // float vertices[] = {
  //     // back face
  //     -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,  // bottom-left
  //     1.0f, 1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f,    // top-right
  //     1.0f, -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f,   // bottom-right
  //     1.0f, 1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f,    // top-right
  //     -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,  // bottom-left
  //     -1.0f, 1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f,   // top-left
  //     // front face
  //     -1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,  // bottom-left
  //     1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f,   // bottom-right
  //     1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,    // top-right
  //     1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,    // top-right
  //     -1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,   // top-left
  //     -1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,  // bottom-left
  //     // left face
  //     -1.0f, 1.0f, 1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f,    // top-right
  //     -1.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f,   // top-left
  //     -1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f,  // bottom-left
  //     -1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f,  // bottom-left
  //     -1.0f, -1.0f, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f,   // bottom-right
  //     -1.0f, 1.0f, 1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f,    // top-right
  //                                                          // right face
  //     1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,      // top-left
  //     1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,    // bottom-right
  //     1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,     // top-right
  //     1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,    // bottom-right
  //     1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,      // top-left
  //     1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,     // bottom-left
  //     // bottom face
  //     -1.0f, -1.0f, -1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f,  // top-right
  //     1.0f, -1.0f, -1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 1.0f,   // top-left
  //     1.0f, -1.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f,    // bottom-left
  //     1.0f, -1.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f,    // bottom-left
  //     -1.0f, -1.0f, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f,   // bottom-right
  //     -1.0f, -1.0f, -1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f,  // top-right
  //     // top face
  //     -1.0f, 1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,  // top-left
  //     1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,    // bottom-right
  //     1.0f, 1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f,   // top-right
  //     1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,    // bottom-right
  //     -1.0f, 1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,  // top-left
  //     -1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f    // bottom-left
  // };
  // postion:vec3, normal:vec3, texcoords:vec2 tangent:vec3 bitangent:vec3
  float vertices[] = {
    // back face
    -1.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, -0.0, -0.0, -0.0, 1.0, -0.0,
    1.0, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 1.0, 1.0, -0.0, -0.0, -0.0, 1.0, -0.0,
    1.0, -1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 1.0, -0.0, -0.0, -0.0, 1.0, -0.0,
    1.0, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 1.0, 1.0, -0.0, -0.0, -0.0, 1.0, -0.0,
    -1.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, -0.0, -0.0, -0.0, 1.0, -0.0,
    -1.0, 1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 1.0, -0.0, -0.0, -0.0, 1.0, -0.0,
    // front face
    -1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -0.0, 0.0, 1.0, 0.0,
    -1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, -0.0, 0.0, 1.0, 0.0,
    -1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, -0.0, 0.0, 1.0, 0.0,
    // left face
    -1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0,
    -1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0,
    -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0,
    -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0,
    -1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0,
    -1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0,
    // right face
    1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, -0.0, 1.0, -0.0, 0.0, -0.0, -1.0,
    1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, -0.0, 1.0, -0.0, 0.0, -0.0, -1.0,
    1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 1.0, -0.0, 1.0, -0.0, 0.0, -0.0, -1.0,
    1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, -0.0, 1.0, -0.0, -0.0, -0.0, -1.0,
    1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, -0.0, 1.0, -0.0, -0.0, -0.0, -1.0,
    1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -0.0, 1.0, -0.0, -0.0, -0.0, -1.0,
    // bottom face
    -1.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -0.0, -0.0, -1.0,
    1.0, -1.0, -1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, -0.0, -0.0, -1.0,
    1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, -0.0, -0.0, -1.0,
    1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -1.0,
    -1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -1.0,
    -1.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, -0.0, -0.0, -0.0, -0.0, -1.0,
    // top face
    -1.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0,
    1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0,
    1.0, 1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0,
    1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -0.0, 0.0, 0.0, -1.0,
    -1.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, -0.0, 0.0, 0.0, -1.0,
    -1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.0, 0.0, 0.0, -1.0,
  };
  unsigned indices[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
  unsigned cubeVAO{0};
  glGenVertexArrays(1, &cubeVAO);
  unsigned cubeVBO{0};
  glGenBuffers(1, &cubeVBO);
  unsigned cubeEBO{0};
  glGenBuffers(1, &cubeEBO);
  // fill buffer
  glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  // link vertex attributes
  glBindVertexArray(cubeVAO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cubeEBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float),
                        (void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 14 * sizeof(float),
                        (void*)(6 * sizeof(float)));
  // vertex tangent
  glEnableVertexAttribArray(3);
  glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float),
                        (void*)(8 * sizeof(float)));
  // vertex bitangent
  glEnableVertexAttribArray(4);
  glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float),
                        (void*)(11 * sizeof(float)));
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
  return SimpleMesh{cubeVAO, sizeof(indices) / sizeof(unsigned)};
}