#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "Scene.h"
#include "Camera.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class RayTracer {
public:
    RayTracer(int width, int height);
    ~RayTracer();

    bool initialize();
    void render(const Camera& camera, const Scene& scene);
    void resize(int width, int height);
    
    unsigned int getTextureID() const { return textureID; }

private:
    int width, height;
    
    // OpenGL resources
    unsigned int textureID;
    unsigned int pbo;
    
    // CUDA resources
    cudaGraphicsResource* cudaPboResource;
    float3* d_output;
    SceneData* d_sceneData;
    
    // Ray tracing parameters
    static const int MAX_DEPTH = 8;
    
    bool initializeGL();
    bool initializeCUDA();
    void cleanup();
};

#endif