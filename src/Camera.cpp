#include "Camera.h"
#include <algorithm>
#include <iostream>

Camera::Camera(Vec3 position, Vec3 target, Vec3 up, float fov, float aspect)
    : position(position), worldUp(up), fov(fov), aspect(aspect),
      speed(5.0f), sensitivity(0.1f), firstMouse(true), lastX(640), lastY(360) {
    
    Vec3 direction = (target - position).normalize();
    yaw = atan2f(direction.z, direction.x) * 180.0f / M_PI;
    pitch = asinf(direction.y) * 180.0f / M_PI;
    
    updateCameraVectors();
    updateProjection();
}

void Camera::update(GLFWwindow* window, float deltaTime) {
    // Handle keyboard input
    float velocity = speed * deltaTime;
    
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        position = position + front * velocity;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        position = position - front * velocity;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        position = position - right * velocity;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        position = position + right * velocity;
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        position = position + worldUp * velocity;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        position = position - worldUp * velocity;

    // Handle mouse input
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // Reversed since y-coordinates go from bottom to top
    lastX = xpos;
    lastY = ypos;

    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    // Constrain pitch
    pitch = std::max(-89.0f, std::min(89.0f, pitch));

    updateCameraVectors();
    updateProjection();
}

Ray Camera::getRay(float u, float v) const {
    Vec3 direction = lowerLeftCorner + horizontal * u + vertical * v - position;
    return Ray(position, direction.normalize());
}

void Camera::updateCameraVectors() {
    // Calculate front vector
    Vec3 newFront;
    newFront.x = cosf(yaw * M_PI / 180.0f) * cosf(pitch * M_PI / 180.0f);
    newFront.y = sinf(pitch * M_PI / 180.0f);
    newFront.z = sinf(yaw * M_PI / 180.0f) * cosf(pitch * M_PI / 180.0f);
    front = newFront.normalize();
    
    // Calculate right and up vectors
    right = front.cross(worldUp).normalize();
    up = right.cross(front).normalize();
}

void Camera::updateProjection() {
    float theta = fov * M_PI / 180.0f;
    float halfHeight = tanf(theta / 2.0f);
    float halfWidth = aspect * halfHeight;
    
    lowerLeftCorner = position - halfWidth * right - halfHeight * up + front;
    horizontal = right * (2.0f * halfWidth);
    vertical = up * (2.0f * halfHeight);
}