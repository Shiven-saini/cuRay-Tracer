/**
 * CUDA Ray Tracing Engine
 * Author: Shiven Saini
 * Email: shiven.career@proton.me
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
const std::string WINDOW_TITLE = "CUDA Ray Tracer - by Shiven Saini";

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "    CUDA Ray Tracing Engine v1.0" << std::endl;
    std::cout << "    Author: Shiven Saini" << std::endl;
    std::cout << "    Email: shiven.career@proton.me" << std::endl;
    std::cout << "    Date: " << __DATE__ << std::endl;
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

    // Initialize camera
    Vec3 cameraPos(0.0f, 2.0f, 8.0f);
    Vec3 target(0.0f, 0.0f, 0.0f);
    Vec3 up(0.0f, 1.0f, 0.0f);
    float fov = 45.0f;
    float aspect = static_cast<float>(WINDOW_WIDTH) / WINDOW_HEIGHT;
    
    auto camera = std::make_unique<Camera>(cameraPos, target, up, fov, aspect);

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

    std::cout << "Starting render loop..." << std::endl;
    std::cout << "You should see: Test pattern first, then ray-traced scene" << std::endl;

    // Set up OpenGL state for modern rendering
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f); // Dark gray background

    // Main render loop
    while (!window->shouldClose()) {
        window->pollEvents();
        fpsCounter->update();
        camera->update(window->getGLFWWindow(), fpsCounter->getDeltaTime());
        
        // Clear screen
        glClear(GL_COLOR_BUFFER_BIT);
        
        // Update ray tracing
        rayTracer->render(*camera, *scene);
        
        // Render the textured quad using modern OpenGL
        rayTracer->renderQuad();
        
        window->swapBuffers();
        
        // Update window title with FPS
        static int frameCounter = 0;
        if (++frameCounter % 60 == 0) {
            std::string title = WINDOW_TITLE + " - " + fpsCounter->getFPSString();
            glfwSetWindowTitle(window->getGLFWWindow(), title.c_str());
        }
    }

    std::cout << "Shutting down..." << std::endl;
    return 0;
}