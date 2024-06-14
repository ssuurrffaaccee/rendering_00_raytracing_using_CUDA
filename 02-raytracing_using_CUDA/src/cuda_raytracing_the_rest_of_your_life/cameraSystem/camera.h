#pragma once
#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Defines several possible options for camera movement. Used as abstraction to
// stay away from window-system specific input methods
enum class Camera_Movement { FORWARD, BACKWARD, LEFT, RIGHT };

// Default camera values
inline const float YAW = 90.4f;
inline const float PITCH = 2.4f;
inline const float SPEED = 2.5f;
inline const float SENSITIVITY = 0.1f;
inline const float ZOOM = 45.0f;

// An abstract camera class that processes input and calculates the
// corresponding Euler Angles, Vectors and Matrices for use in OpenGL
class Camera {
 public:
  // camera Attributes
  glm::vec3 position_;
  glm::vec3 front_;
  glm::vec3 up_;
  glm::vec3 right_;
  glm::vec3 worldUp_;
  // euler Angles
  float yaw_;
  float pitch_;
  // camera options
  float movementSpeed_;
  float mouseSensitivity_;
  float zoom_;
  bool dirty_{true};

  // constructor with vectors
  Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f),
         glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = YAW,
         float pitch = PITCH);
  // constructor with scalar values
  Camera(float posX, float posY, float posZ, float upX, float upY, float upZ,
         float yaw, float pitch);

  // returns the view matrix calculated using Euler Angles and the LookAt Matrix
  glm::mat4 getViewMatrix();

  // processes input received from any keyboard-like input system. Accepts input
  // parameter in the form of camera defined ENUM (to abstract it from windowing
  // systems)
  void processKeyboard(Camera_Movement direction, float deltaTime);

  // processes input received from a mouse input system. Expects the offset
  // value in both the x and y direction.
  void processMouseMovement(float xoffset, float yoffset,
                            GLboolean constrainpitch_ = true);

  // processes input received from a mouse scroll-wheel event. Only requires
  // input on the vertical wheel-axis
  void processMouseScroll(float yoffset);

 private:
  // calculates the front vector from the Camera's (updated) Euler Angles
  void updateCameraVectors();
};