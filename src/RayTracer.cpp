#include <GL/glew.h>  // MUST be included FIRST
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
    // Check OpenGL version
    const char* version = (const char*)glGetString(GL_VERSION);
    std::cout << "OpenGL Version: " << (version ? version : "Unknown") << std::endl;
    
    // Create texture with correct format
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
    
    // Use GL_RGBA32F for internal format and GL_RGBA + GL_FLOAT for data format
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "Failed to create texture, error: " << err << std::endl;
        return false;
    }
    std::cout << "Texture created: " << textureID << " (" << width << "x" << height << ")" << std::endl;
    
    // Create PBO with the correct size for RGBA float data
    glGenBuffers(1, &pbo);
    if (glGetError() != GL_NO_ERROR) {
        std::cerr << "Failed to generate PBO" << std::endl;
        return false;
    }
    
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "Failed to create PBO, error: " << err << std::endl;
        return false;
    }
    std::cout << "PBO created: " << pbo << " (size: " << (width * height * 4 * sizeof(float)) << " bytes)" << std::endl;
    
    return true;
}

bool RayTracer::initializeCUDA() {
    // Register PBO with CUDA
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cerr << "Failed to register PBO with CUDA: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    std::cout << "PBO registered with CUDA successfully" << std::endl;
    
    // Allocate device memory for scene data
    err = cudaMalloc(&d_sceneData, sizeof(SceneData));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    std::cout << "Device memory allocated: " << sizeof(SceneData) << " bytes" << std::endl;
    
    return true;
}

void RayTracer::render(const Camera& camera, const Scene& scene) {
    static int frameCount = 0;
    frameCount++;
    
    if (frameCount % 60 == 0) {
        std::cout << "Rendering frame " << frameCount << "..." << std::endl;
    }
    
    // Copy scene data to device
    cudaError_t err = cudaMemcpy(d_sceneData, &scene.getSceneData(), sizeof(SceneData), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy scene data: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Map PBO resource
    err = cudaGraphicsMapResources(1, &cudaPboResource, 0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to map PBO resource: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    size_t numBytes;
    err = cudaGraphicsResourceGetMappedPointer((void**)&d_output, &numBytes, cudaPboResource);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get mapped pointer: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    if (frameCount % 60 == 0) {
        std::cout << "Mapped " << numBytes << " bytes from PBO" << std::endl;
    }
    
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
    
    // Cast to float4* for CUDA kernel (RGBA format)
    launchRayTracingKernel((float4*)d_output, width, height, d_sceneData,
                          cameraPos, cameraDir, cameraUp, cameraRight, 45.0f);
    
    CudaUtils::checkKernelLaunch("rayTracingKernel");
    
    // Unmap PBO resource
    err = cudaGraphicsUnmapResources(1, &cudaPboResource, 0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to unmap PBO resource: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Update texture from PBO
    glBindTexture(GL_TEXTURE_2D, textureID);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    
    GLenum glErr = glGetError();
    if (glErr != GL_NO_ERROR) {
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