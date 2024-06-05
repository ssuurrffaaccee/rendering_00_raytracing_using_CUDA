#include "camera.h"
#include <iostream>
// constructor with vectors
Camera::Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch)
    : front_(glm::vec3{0.0f} - position),
      movementSpeed_(SPEED),
      mouseSensitivity_(SENSITIVITY),
      zoom_(ZOOM) {
  position_ = position;
  worldUp_ = up;
  yaw_ = yaw;
  pitch_ = pitch;
  updateCameraVectors();
}
// constructor with scalar values
Camera::Camera(float posX, float posY, float posZ, float upX, float upY,
               float upZ, float yaw, float pitch)
    : front_(glm::vec3(0.0f, 0.0f, 1.0f)),
      movementSpeed_(SPEED),
      mouseSensitivity_(SENSITIVITY),
      zoom_(ZOOM) {
  position_ = glm::vec3(posX, posY, posZ);
  worldUp_ = glm::vec3(upX, upY, upZ);
  yaw_ = yaw;
  pitch_ = pitch;
  updateCameraVectors();
}

// returns the view matrix calculated using Euler Angles and the LookAt Matrix
glm::mat4 Camera::getViewMatrix() {
  return glm::lookAt(position_, position_ + front_, up_);
}

// processes input received from any keyboard-like input system. Accepts input
// parameter in the form of camera defined ENUM (to abstract it from windowing
// systems)
void Camera::processKeyboard(Camera_Movement direction, float deltaTime) {
  float velocity = movementSpeed_ * deltaTime;
  if (direction == Camera_Movement::FORWARD) position_ += front_ * velocity;
  if (direction == Camera_Movement::BACKWARD) position_ -= front_ * velocity;
  if (direction == Camera_Movement::LEFT) position_ -= right_ * velocity;
  if (direction == Camera_Movement::RIGHT) position_ += right_ * velocity;
  dirty_ = true;
}

// processes input received from a mouse input system. Expects the offset value
// in both the x and y direction.
void Camera::processMouseMovement(float xoffset, float yoffset,
                                  GLboolean constrainpitch_) {
  xoffset *= mouseSensitivity_;
  yoffset *= mouseSensitivity_;

  yaw_ += xoffset;
  pitch_ += yoffset;

  // make sure that when pitch is out of bounds, screen doesn't get flipped
  if (constrainpitch_) {
    if (pitch_ > 89.0f) pitch_ = 89.0f;
    if (pitch_ < -89.0f) pitch_ = -89.0f;
  }

  // update front_, right_ and Up Vectors using the updated Euler angles
  updateCameraVectors();
  dirty_ = true;
  // std::cout<<yaw_<<" "<<pitch_<<"\n";
}

// processes input received from a mouse scroll-wheel event. Only requires input
// on the vertical wheel-axis
void Camera::processMouseScroll(float yoffset) {
  zoom_ -= (float)yoffset;
  if (zoom_ < 1.0f) zoom_ = 1.0f;
  if (zoom_ > 45.0f) zoom_ = 45.0f;
  dirty_ = true;
}

// calculates the front vector from the Camera's (updated) Euler Angles
void Camera::updateCameraVectors() {
  // calculate the new front_ vector
  glm::vec3 front;
  front.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
  front.y = sin(glm::radians(pitch_));
  front.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
  front_ = glm::normalize(front);
  // also re-calculate the right_ and Up vector
  right_ = glm::normalize(glm::cross(
      front_, worldUp_));  // normalize the vectors, because their length gets
                           // closer to 0 the more you look up or down which
                           // results in slower movement.
  up_ = glm::normalize(glm::cross(right_, front_));
  dirty_ = true;
}