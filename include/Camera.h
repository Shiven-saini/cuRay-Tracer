#ifndef CAMERA_H
#define CAMERA_H

#include "math/Vec3.h"
#include "math/Ray.h"

// Forward declare GLFW to avoid header conflicts
struct GLFWwindow;

class Camera {
public:
    Camera(Vec3 position, Vec3 target, Vec3 up, float fov, float aspect);
    ~Camera() = default;

    void update(GLFWwindow* window, float deltaTime);
    Ray getRay(float u, float v) const;
    
    Vec3 getPosition() const { return position; }
    Vec3 getDirection() const { return front; }

    // New method for orbit camera
    void setOrbitMode(bool enabled) { orbitMode = enabled; }
    void setTarget(const Vec3& newTarget) { target = newTarget; updateOrbitCamera(); }

private:
    Vec3 position;
    Vec3 front;
    Vec3 up;
    Vec3 right;
    Vec3 worldUp;
    Vec3 target; // Target point for orbit camera

    float yaw;
    float pitch;
    float fov;
    float aspect;
    float speed;
    float sensitivity;

    // Orbit camera parameters
    bool orbitMode;
    float orbitDistance;
    float orbitYaw;
    float orbitPitch;

    Vec3 lowerLeftCorner;
    Vec3 horizontal;
    Vec3 vertical;

    bool firstMouse;
    float lastX, lastY;

    void updateCameraVectors();
    void updateProjection();
    void updateOrbitCamera();
    static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
};

#endif