#include <GL/glew.h> 
#include "RayTracer.h"
#include "cuda/raytracing_kernel.h"
#include "cuda/cuda_utils.h"
#include <iostream>

RayTracer::RayTracer(int width, int height)
    : width(width), height(height), textureID(0), pbo(0),
      cudaPboResource(nullptr), d_output(nullptr), d_sceneData(nullptr) {}

RayTracer::~RayTracer() {
    cleanup();
}

bool RayTracer::initialize() {
    if (!initializeGL()) {
        std::cerr << "Failed to initialize OpenGL resources" << std::endl;
        return false;
    }
    
    if (!initializeCUDA()) {
        std::cerr << "Failed to initialize CUDA resources" << std::endl;
        return false;
    }
    
    return true;
}

bool RayTracer::initializeGL() {
    // Create texture
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    
    // Create PBO
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    
    return glGetError() == GL_NO_ERROR;
}

bool RayTracer::initializeCUDA() {
    // Register PBO with CUDA
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo, cudaGraphicsMapFlagsWriteDiscard));
    
    // Allocate device memory for scene data
    CUDA_CHECK(cudaMalloc(&d_sceneData, sizeof(SceneData)));
    
    return true;
}

void RayTracer::render(const Camera& camera, const Scene& scene) {
    // Copy scene data to device
    CUDA_CHECK(cudaMemcpy(d_sceneData, &scene.getSceneData(), sizeof(SceneData), cudaMemcpyHostToDevice));
    
    // Map PBO resource
    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPboResource, 0));
    
    size_t numBytes;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_output, &numBytes, cudaPboResource));
    
    // Launch ray tracing kernel
    Vec3 cameraPos = camera.getPosition();
    Vec3 cameraDir = camera.getDirection();
    Vec3 cameraUp(0, 1, 0);
    Vec3 cameraRight = cameraDir.cross(cameraUp).normalize();
    cameraUp = cameraRight.cross(cameraDir).normalize();
    
    launchRayTracingKernel(d_output, width, height, d_sceneData,
                          cameraPos, cameraDir, cameraUp, cameraRight, 45.0f);
    
    // Unmap PBO resource
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPboResource, 0));
    
    // Update texture from PBO
    glBindTexture(GL_TEXTURE_2D, textureID);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void RayTracer::resize(int newWidth, int newHeight) {
    width = newWidth;
    height = newHeight;
    
    cleanup();
    initialize();
}

void RayTracer::cleanup() {
    if (cudaPboResource) {
        cudaGraphicsUnregisterResource(cudaPboResource);
        cudaPboResource = nullptr;
    }
    
    if (d_sceneData) {
        cudaFree(d_sceneData);
        d_sceneData = nullptr;
    }
    
    if (pbo) {
        glDeleteBuffers(1, &pbo);
        pbo = 0;
    }
    
    if (textureID) {
        glDeleteTextures(1, &textureID);
        textureID = 0;
    }
}