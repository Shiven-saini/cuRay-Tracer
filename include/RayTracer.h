#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "Scene.h"
#include "Camera.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GL/glew.h>

class RayTracer {
public:
    RayTracer(int width, int height);
    ~RayTracer();

    bool initialize();
    void render(const Camera& camera, const Scene& scene);
    void resize(int newWidth, int newHeight);
    
    unsigned int getTextureID() const { return textureID; }
    void renderQuad(); // Add this method

private:
    int width, height;
    
    // OpenGL resources
    unsigned int textureID;
    unsigned int pbo;
    unsigned int shaderProgram;
    unsigned int VAO, VBO, EBO;
    
    // CUDA resources
    cudaGraphicsResource* cudaPboResource;
    float3* d_output;
    SceneData* d_sceneData;
    
    // Ray tracing parameters
    static const int MAX_DEPTH = 8;
    
    bool initializeGL();
    bool initializeCUDA();
    bool createShaderProgram();
    bool createQuadGeometry();
    void cleanup();
};

#endif