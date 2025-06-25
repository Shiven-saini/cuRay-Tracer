#include <GL/glew.h>  // MUST be included FIRST
#include "RayTracer.h"
#include "cuda/raytracing_kernel.h"
#include "cuda/cuda_utils.h"
#include <iostream>
#include <vector>

RayTracer::RayTracer(int width, int height)
    : width(width), height(height), textureID(0), pbo(0),
      cudaPboResource(nullptr), d_output(nullptr), d_sceneData(nullptr) {}

RayTracer::~RayTracer() {
    cleanup();
}

bool RayTracer::initialize() {
    std::cout << "Initializing RayTracer..." << std::endl;
    
    if (!initializeGL()) {
        std::cerr << "Failed to initialize OpenGL resources" << std::endl;
        return false;
    }
    std::cout << "OpenGL resources initialized successfully" << std::endl;
    
    if (!initializeCUDA()) {
        std::cerr << "Failed to initialize CUDA resources" << std::endl;
        return false;
    }
    std::cout << "CUDA resources initialized successfully" << std::endl;
    
    return true;
}

bool RayTracer::initializeGL() {
    // Check OpenGL version and extensions
    const char* version = (const char*)glGetString(GL_VERSION);
    const char* vendor = (const char*)glGetString(GL_VENDOR);
    const char* renderer = (const char*)glGetString(GL_RENDERER);
    
    std::cout << "OpenGL Version: " << (version ? version : "Unknown") << std::endl;
    std::cout << "OpenGL Vendor: " << (vendor ? vendor : "Unknown") << std::endl;
    std::cout << "OpenGL Renderer: " << (renderer ? renderer : "Unknown") << std::endl;
    
    // Create texture with standard RGB format first
    glGenTextures(1, &textureID);
    if (glGetError() != GL_NO_ERROR) {
        std::cerr << "Failed to generate texture" << std::endl;
        return false;
    }
    
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // Try simpler RGB format instead of RGBA32F
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "Failed to create texture, error: " << err << std::endl;
        return false;
    }
    std::cout << "Texture created: " << textureID << " (" << width << "x" << height << ") - RGB format" << std::endl;
    
    // Don't use PBO for now - direct memory approach
    glBindTexture(GL_TEXTURE_2D, 0);
    
    return true;
}

bool RayTracer::initializeCUDA() {
    // Allocate device memory for output (RGB format)
    CUDA_CHECK(cudaMalloc(&d_output, width * height * 3 * sizeof(unsigned char)));
    
    // Allocate device memory for scene data
    CUDA_CHECK(cudaMalloc(&d_sceneData, sizeof(SceneData)));
    
    std::cout << "CUDA memory allocated successfully" << std::endl;
    
    return true;
}

void RayTracer::render(const Camera& camera, const Scene& scene) {
    static int frameCount = 0;
    frameCount++;
    
    if (frameCount % 60 == 0) {
        std::cout << "Rendering frame " << frameCount << "..." << std::endl;
    }
    
    // Copy scene data to device
    CUDA_CHECK(cudaMemcpy(d_sceneData, &scene.getSceneData(), sizeof(SceneData), cudaMemcpyHostToDevice));
    
    // Launch ray tracing kernel
    Vec3 cameraPos = camera.getPosition();
    Vec3 cameraDir = camera.getDirection();
    Vec3 cameraUp(0, 1, 0);
    Vec3 cameraRight = cameraDir.cross(cameraUp).normalize();
    cameraUp = cameraRight.cross(cameraDir).normalize();
    
    if (frameCount % 60 == 0) {
        std::cout << "Camera pos: (" << cameraPos.x << ", " << cameraPos.y << ", " << cameraPos.z << ")" << std::endl;
        std::cout << "Camera dir: (" << cameraDir.x << ", " << cameraDir.y << ", " << cameraDir.z << ")" << std::endl;
    }
    
    // Launch kernel with unsigned char output
    launchRayTracingKernel((unsigned char*)d_output, width, height, d_sceneData,
                          cameraPos, cameraDir, cameraUp, cameraRight, 45.0f);
    
    CudaUtils::checkKernelLaunch("rayTracingKernel");
    
    // Copy result back to host
    static std::vector<unsigned char> hostOutput(width * height * 3);
    CUDA_CHECK(cudaMemcpy(hostOutput.data(), d_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    
    // Update texture directly
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, hostOutput.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    
    GLenum glErr = glGetError();
    if (glErr != GL_NO_ERROR && frameCount % 60 == 0) {
        std::cerr << "OpenGL error during texture update: " << glErr << std::endl;
    }
}

void RayTracer::resize(int newWidth, int newHeight) {
    width = newWidth;
    height = newHeight;
    
    cleanup();
    initialize();
}

void RayTracer::cleanup() {
    if (d_output) {
        cudaFree(d_output);
        d_output = nullptr;
    }
    
    if (d_sceneData) {
        cudaFree(d_sceneData);
        d_sceneData = nullptr;
    }
    
    if (textureID) {
        glDeleteTextures(1, &textureID);
        textureID = 0;
    }
}