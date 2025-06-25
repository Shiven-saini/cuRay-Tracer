#include "cuda/raytracing_kernel.h"
#include "math/Vec3.h"
#include "math/Ray.h"
#include "primitives/Sphere.h"
#include "primitives/Plane.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_RAY_DEPTH 4
#define EPSILON 0.001f

__device__ Vec3 reflect(const Vec3& incident, const Vec3& normal) {
    return incident - normal * (2.0f * incident.dot(normal));
}

__device__ bool intersectScene(const Ray& ray, const SceneData* scene, float& minT, 
                              Vec3& hitPoint, Vec3& normal, Material& material) {
    minT = FLT_MAX;
    bool hit = false;
    
    // Check sphere intersections
    for (int i = 0; i < scene->numSpheres && i < 10; i++) {
        float t;
        Vec3 n;
        if (scene->spheres[i].intersect(ray, t, n) && t < minT && t > EPSILON) {
            minT = t;
            normal = n;
            material = scene->spheres[i].material;
            hit = true;
        }
    }
    
    // Check plane intersections
    for (int i = 0; i < scene->numPlanes && i < 10; i++) {
        float t;
        Vec3 n;
        if (scene->planes[i].intersect(ray, t, n) && t < minT && t > EPSILON) {
            minT = t;
            normal = n;
            material = scene->planes[i].material;
            hit = true;
        }
    }
    
    if (hit) {
        hitPoint = ray.at(minT);
    }
    
    return hit;
}

__device__ Vec3 calculateLighting(const Vec3& point, const Vec3& normal, const Vec3& viewDir,
                                 const Material& material, const SceneData* scene) {
    Vec3 color = material.color * 0.2f; // Ambient
    
    // Simple diffuse lighting
    Vec3 lightDir = (scene->lightPosition - point).normalize();
    float diff = fmaxf(0.0f, normal.dot(lightDir));
    color = color + material.color * diff * 0.8f;
    
    return color;
}

__device__ Vec3 traceRay(Ray ray, const SceneData* scene, int depth = 0) {
    if (depth >= MAX_RAY_DEPTH) {
        return Vec3(0, 0, 0);
    }
    
    float t;
    Vec3 hitPoint, normal;
    Material material;
    
    if (!intersectScene(ray, scene, t, hitPoint, normal, material)) {
        // Sky gradient
        float y = fmaxf(0.0f, ray.direction.y);
        return Vec3(0.3f + 0.7f * y, 0.5f + 0.5f * y, 0.7f + 0.3f * y);
    }
    
    Vec3 viewDir = (ray.origin - hitPoint).normalize();
    Vec3 color = calculateLighting(hitPoint, normal, viewDir, material, scene);
    
    // Simple reflection for reflective materials
    if (material.type == REFLECTIVE && depth < MAX_RAY_DEPTH - 1) {
        Vec3 reflectDir = reflect(ray.direction, normal);
        Ray reflectRay(hitPoint + reflectDir * EPSILON, reflectDir);
        Vec3 reflectColor = traceRay(reflectRay, scene, depth + 1);
        color = color * (1.0f - material.reflectivity) + reflectColor * material.reflectivity;
    }
    
    return color;
}

__global__ void rayTracingKernel(unsigned char* output, int width, int height,
                                const SceneData* scene, Vec3 cameraPos, 
                                Vec3 cameraFront, Vec3 cameraUp, Vec3 cameraRight,
                                float fov) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * 3;
    
    // Large test area to make sure we can see it
    if (x < 100 && y < 100) {
        output[idx + 0] = 255; // Bright red
        output[idx + 1] = 0;
        output[idx + 2] = 0;
        return;
    }
    
    if (x >= width - 100 && y < 100) {
        output[idx + 0] = 0;
        output[idx + 1] = 255; // Bright green
        output[idx + 2] = 0;
        return;
    }
    
    if (x < 100 && y >= height - 100) {
        output[idx + 0] = 0;
        output[idx + 1] = 0;
        output[idx + 2] = 255; // Bright blue
        return;
    }
    
    if (x >= width - 100 && y >= height - 100) {
        output[idx + 0] = 255;
        output[idx + 1] = 255; // Bright yellow
        output[idx + 2] = 0;
        return;
    }
    
    // Ray tracing for the rest
    float aspectRatio = float(width) / float(height);
    float theta = fov * M_PI / 180.0f;
    float halfHeight = tanf(theta / 2.0f);
    float halfWidth = aspectRatio * halfHeight;
    
    float u = (float(x) + 0.5f) / float(width);
    float v = 1.0f - (float(y) + 0.5f) / float(height);
    
    u = (u - 0.5f) * 2.0f * halfWidth;
    v = (v - 0.5f) * 2.0f * halfHeight;
    
    Vec3 rayDir = (cameraFront + cameraRight * u + cameraUp * v).normalize();
    Ray ray(cameraPos, rayDir);
    
    Vec3 color = traceRay(ray, scene);
    
    // Clamp and convert to bytes
    color.x = fminf(fmaxf(color.x, 0.0f), 1.0f);
    color.y = fminf(fmaxf(color.y, 0.0f), 1.0f);
    color.z = fminf(fmaxf(color.z, 0.0f), 1.0f);
    
    output[idx + 0] = (unsigned char)(color.x * 255.0f);
    output[idx + 1] = (unsigned char)(color.y * 255.0f);
    output[idx + 2] = (unsigned char)(color.z * 255.0f);
}

extern "C" void launchRayTracingKernel(unsigned char* output, int width, int height,
                                      const SceneData* scene, Vec3 cameraPos, 
                                      Vec3 cameraFront, Vec3 cameraUp, Vec3 cameraRight,
                                      float fov, cudaStream_t stream) {
    // Clear output first
    cudaMemset(output, 128, width * height * 3); // Gray background
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    rayTracingKernel<<<gridSize, blockSize, 0, stream>>>(
        output, width, height, scene, cameraPos, cameraFront, cameraUp, cameraRight, fov
    );
    
    cudaDeviceSynchronize();
}