/**
 * CUDA Ray Tracing Engine v1.0
 * Author: Shiven Saini
 * Email: shiven.career@proton.me
 * Date: 2025-01-26
 * 
 * Interactive Controls:
 * - Mouse: Orbit camera around scene
 * - WASD: Move light source
 * - ESC: Exit
 */

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <memory>

#include "Window.h"
#include "Camera.h"
#include "Scene.h"
#include "RayTracer.h"
#include "FPSCounter.h"
#include "cuda/cuda_utils.h"

const int WINDOW_WIDTH = 1280;
const int WINDOW_HEIGHT = 720;
const std::string WINDOW_TITLE = "CUDA Ray Tracer - Interactive Scene";

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CUDA Ray Tracing Engine v1.0" << std::endl;
    std::cout << "  Author: Shiven Saini" << std::endl;
    std::cout << "  Email: shiven.career@proton.me" << std::endl;
    std::cout << "  Date: 2025-01-26" << std::endl;
    std::cout << "========================================" << std::endl;

    // Check CUDA capabilities
    if (!CudaUtils::checkCudaCapabilities()) {
        std::cerr << "CUDA requirements not met!" << std::endl;
        return -1;
    }

    CudaUtils::printDeviceInfo();

    // Initialize window
    auto window = std::make_unique<Window>(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE);
    if (!window->initialize()) {
        std::cerr << "Failed to initialize window!" << std::endl;
        return -1;
    }

    // Initialize camera in orbit mode
    Vec3 target(0.0f, 1.5f, 0.0f);  // Look at center of scene
    Vec3 cameraPos(0.0f, 3.0f, 8.0f);
    Vec3 up(0.0f, 1.0f, 0.0f);
    float fov = 45.0f;
    float aspect = static_cast<float>(WINDOW_WIDTH) / WINDOW_HEIGHT;
    
    auto camera = std::make_unique<Camera>(cameraPos, target, up, fov, aspect);
    camera->setOrbitMode(true);

    // Initialize scene
    auto scene = std::make_unique<Scene>();
    scene->setupRoomScene();

    // Initialize ray tracer
    auto rayTracer = std::make_unique<RayTracer>(WINDOW_WIDTH, WINDOW_HEIGHT);
    if (!rayTracer->initialize()) {
        std::cerr << "Failed to initialize ray tracer!" << std::endl;
        return -1;
    }

    // Initialize FPS counter
    auto fpsCounter = std::make_unique<FPSCounter>();

    std::cout << "\n========================================" << std::endl;
    std::cout << "        INTERACTIVE CONTROLS" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Mouse Movement  : Orbit camera around scene" << std::endl;
    std::cout << "W / S Keys      : Move light forward/back" << std::endl;
    std::cout << "A / D Keys      : Move light left/right" << std::endl;
    std::cout << "ESC Key         : Exit application" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Starting interactive ray tracing..." << std::endl;

    // Set up OpenGL state
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    // Main render loop
    while (!window->shouldClose()) {
        window->pollEvents();
        fpsCounter->update();
        
        // Update camera (mouse orbit)
        camera->update(window->getGLFWWindow(), fpsCounter->getDeltaTime());
        
        // Update light position based on WASD keys
        bool moveUp = glfwGetKey(window->getGLFWWindow(), GLFW_KEY_W) == GLFW_PRESS;
        bool moveDown = glfwGetKey(window->getGLFWWindow(), GLFW_KEY_S) == GLFW_PRESS;
        bool moveLeft = glfwGetKey(window->getGLFWWindow(), GLFW_KEY_A) == GLFW_PRESS;
        bool moveRight = glfwGetKey(window->getGLFWWindow(), GLFW_KEY_D) == GLFW_PRESS;
        
        scene->updateLight(fpsCounter->getDeltaTime(), moveUp, moveDown, moveLeft, moveRight);
        
        // Display current light position occasionally
        static int frameCounter = 0;
        if (++frameCounter % 300 == 0) { // Every 5 seconds at 60fps
            Vec3 lightPos = scene->getLightPosition();
            std::cout << "Light position: (" << lightPos.x << ", " << lightPos.y << ", " << lightPos.z << ")" << std::endl;
        }
        
        // Clear screen
        glClear(GL_COLOR_BUFFER_BIT);
        
        // Update ray tracing
        rayTracer->render(*camera, *scene);
        
        // Render the textured quad
        rayTracer->renderQuad();
        
        window->swapBuffers();
        
        // Update window title with FPS and light position
        if (frameCounter % 60 == 0) {
            Vec3 lightPos = scene->getLightPosition();
            std::string title = WINDOW_TITLE + " - " + fpsCounter->getFPSString() + 
                              " - Light(" + std::to_string((int)lightPos.x) + "," + 
                              std::to_string((int)lightPos.y) + "," + std::to_string((int)lightPos.z) + ")";
            glfwSetWindowTitle(window->getGLFWWindow(), title.c_str());
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Thanks for using CUDA Ray Tracer!" << std::endl;
    std::cout << "  Author: Shiven Saini" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}