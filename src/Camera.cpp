#include "Camera.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <iostream>

Camera::Camera(Vec3 position, Vec3 target, Vec3 up, float fov, float aspect)
    : position(position), target(target), worldUp(up), fov(fov), aspect(aspect),
      speed(5.0f), sensitivity(0.1f), firstMouse(true), lastX(640), lastY(360),
      orbitMode(true), orbitDistance(8.0f), orbitYaw(0.0f), orbitPitch(20.0f) {
    
    // Calculate initial orbit distance
    orbitDistance = (position - target).length();
    
    // Calculate initial orbit angles
    Vec3 direction = (position - target).normalize();
    orbitYaw = atan2f(direction.x, direction.z) * 180.0f / M_PI;
    orbitPitch = asinf(direction.y) * 180.0f / M_PI;
    
    updateOrbitCamera();
    updateProjection();
}

void Camera::update(GLFWwindow* window, float deltaTime) {
    // Handle mouse input for camera orbit
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

    // Only update camera if mouse button is pressed or in continuous mode
    bool mousePressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    
    if (mousePressed || true) { // Always allow mouse control for smooth experience
        xoffset *= sensitivity;
        yoffset *= sensitivity;

        if (orbitMode) {
            orbitYaw += xoffset;
            orbitPitch += yoffset;

            // Constrain pitch
            orbitPitch = std::max(-89.0f, std::min(89.0f, orbitPitch));

            updateOrbitCamera();
        } else {
            yaw += xoffset;
            pitch += yoffset;
            pitch = std::max(-89.0f, std::min(89.0f, pitch));
            updateCameraVectors();
        }
    }

    // Handle scroll for zoom (optional)
    // Note: You'd need to set up a scroll callback for this to work
    
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

void Camera::updateOrbitCamera() {
    // Convert spherical coordinates to Cartesian
    float yawRad = orbitYaw * M_PI / 180.0f;
    float pitchRad = orbitPitch * M_PI / 180.0f;
    
    position.x = target.x + orbitDistance * sinf(yawRad) * cosf(pitchRad);
    position.y = target.y + orbitDistance * sinf(pitchRad);
    position.z = target.z + orbitDistance * cosf(yawRad) * cosf(pitchRad);
    
    // Update camera vectors to look at target
    front = (target - position).normalize();
    right = front.cross(worldUp).normalize();
    up = right.cross(front).normalize();
}

void Camera::updateProjection() {
    float theta = fov * M_PI / 180.0f;
    float halfHeight = tanf(theta / 2.0f);
    float halfWidth = aspect * halfHeight;
    
    lowerLeftCorner = position - right * halfWidth - up * halfHeight + front;
    horizontal = right * (2.0f * halfWidth);
    vertical = up * (2.0f * halfHeight);
}