#include "cuda/raytracing_kernel.h"
#include "math/Vec3.h"
#include "math/Ray.h"
#include "primitives/Sphere.h"
#include "primitives/Plane.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <cfloat>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_RAY_DEPTH 8
#define EPSILON 0.001f

__device__ Vec3 reflect(const Vec3& incident, const Vec3& normal) {
    return incident - normal * (2.0f * incident.dot(normal));
}

__device__ Vec3 refract(const Vec3& incident, const Vec3& normal, float eta) {
    float cosI = -incident.dot(normal);
    float sinT2 = eta * eta * (1.0f - cosI * cosI);
    if (sinT2 >= 1.0f) return Vec3(0, 0, 0); // Total internal reflection
    return incident * eta + normal * (eta * cosI - sqrtf(1.0f - sinT2));
}

__device__ bool intersectScene(const Ray& ray, const SceneData* scene, float& minT, 
                              Vec3& hitPoint, Vec3& normal, Material& material) {
    minT = FLT_MAX;
    bool hit = false;
    
    // Check sphere intersections
    for (int i = 0; i < scene->numSpheres; i++) {
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
    for (int i = 0; i < scene->numPlanes; i++) {
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

__device__ bool isInShadow(const Vec3& point, const Vec3& lightPos, const SceneData* scene) {
    Vec3 lightDir = (lightPos - point).normalize();
    Ray shadowRay(point + lightDir * EPSILON, lightDir);
    
    float lightDistance = (lightPos - point).length();
    
    // Check if any object blocks the light
    for (int i = 0; i < scene->numSpheres; i++) {
        float t;
        Vec3 normal;
        if (scene->spheres[i].intersect(shadowRay, t, normal) && t < lightDistance) {
            return true;
        }
    }
    
    for (int i = 0; i < scene->numPlanes; i++) {
        float t;
        Vec3 normal;
        if (scene->planes[i].intersect(shadowRay, t, normal) && t < lightDistance) {
            return true;
        }
    }
    
    return false;
}

__device__ Vec3 calculateLighting(const Vec3& point, const Vec3& normal, const Vec3& viewDir,
                                 const Material& material, const SceneData* scene) {
    Vec3 color(0, 0, 0);
    
    // Ambient lighting
    Vec3 ambient = material.color * 0.1f;
    color = color + ambient;
    
    // Direct lighting
    Vec3 lightDir = (scene->lightPosition - point).normalize();
    float lightDistance = (scene->lightPosition - point).length();
    
    if (!isInShadow(point, scene->lightPosition, scene)) {
        // Diffuse lighting
        float diff = fmaxf(0.0f, normal.dot(lightDir));
        Vec3 diffuse = material.color * diff * scene->lightColor;
        
        // Specular lighting (Blinn-Phong)
        Vec3 halfDir = (lightDir + viewDir).normalize();
        float spec = powf(fmaxf(0.0f, normal.dot(halfDir)), 32.0f);
        Vec3 specular = Vec3(1, 1, 1) * spec * 0.3f;
        
        // Attenuation
        float attenuation = 1.0f / (1.0f + 0.1f * lightDistance + 0.01f * lightDistance * lightDistance);
        
        color = color + (diffuse + specular) * attenuation;
    }
    
    return color;
}

__device__ Vec3 traceRay(Ray ray, const SceneData* scene, curandState* randState, int depth = 0) {
    if (depth >= MAX_RAY_DEPTH) {
        return Vec3(0, 0, 0);
    }
    
    float t;
    Vec3 hitPoint, normal;
    Material material;
    
    if (!intersectScene(ray, scene, t, hitPoint, normal, material)) {
        // Sky color gradient
        float y = ray.direction.y;
        Vec3 skyColor = Vec3(0.5f, 0.7f, 1.0f) * (0.5f + 0.5f * y) + Vec3(1.0f, 0.9f, 0.8f) * (0.5f - 0.5f * y);
        return skyColor;
    }
    
    Vec3 viewDir = (ray.origin - hitPoint).normalize();
    Vec3 color(0, 0, 0);
    
    switch (material.type) {
        case DIFFUSE: {
            color = calculateLighting(hitPoint, normal, viewDir, material, scene);
            break;
        }
        
        case REFLECTIVE: {
            Vec3 baseColor = calculateLighting(hitPoint, normal, viewDir, material, scene);
            
            Vec3 reflectDir = reflect(ray.direction, normal);
            Ray reflectRay(hitPoint + reflectDir * EPSILON, reflectDir);
            Vec3 reflectColor = traceRay(reflectRay, scene, randState, depth + 1);
            
            color = baseColor * (1.0f - material.reflectivity) + reflectColor * material.reflectivity;
            break;
        }
        
        case REFRACTIVE: {
            float fresnel = material.reflectivity;
            bool entering = ray.direction.dot(normal) < 0;
            Vec3 n = entering ? normal : normal * -1.0f;
            float eta = entering ? 1.0f / material.refractiveIndex : material.refractiveIndex;
            
            Vec3 refractDir = refract(ray.direction, n, eta);
            
            if (refractDir.lengthSquared() > 0) {
                // Refraction
                Ray refractRay(hitPoint + refractDir * EPSILON, refractDir);
                Vec3 refractColor = traceRay(refractRay, scene, randState, depth + 1);
                
                // Reflection
                Vec3 reflectDir = reflect(ray.direction, n);
                Ray reflectRay(hitPoint + reflectDir * EPSILON, reflectDir);
                Vec3 reflectColor = traceRay(reflectRay, scene, randState, depth + 1);
                
                color = refractColor * (1.0f - fresnel) + reflectColor * fresnel;
            } else {
                // Total internal reflection
                Vec3 reflectDir = reflect(ray.direction, n);
                Ray reflectRay(hitPoint + reflectDir * EPSILON, reflectDir);
                color = traceRay(reflectRay, scene, randState, depth + 1);
            }
            break;
        }
        
        case EMISSIVE: {
            color = material.color;
            break;
        }
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
    
    int idx = (y * width + x) * 3; // RGB format
    
    // Simple test: output red color for first few pixels to verify kernel is running
    if (x < 10 && y < 10) {
        output[idx + 0] = 255; // R
        output[idx + 1] = 0;   // G
        output[idx + 2] = 0;   // B
        return;
    }
    
    // Initialize random state
    curandState randState;
    curand_init(x + y * width, 0, 0, &randState);
    
    // Calculate ray direction
    float aspectRatio = float(width) / float(height);
    float theta = fov * M_PI / 180.0f;
    float halfHeight = tanf(theta / 2.0f);
    float halfWidth = aspectRatio * halfHeight;
    
    float u = (float(x) + 0.5f) / float(width);
    float v = (float(y) + 0.5f) / float(height);
    
    u = (u - 0.5f) * 2.0f * halfWidth;
    v = (v - 0.5f) * 2.0f * halfHeight;
    
    Vec3 rayDir = (cameraFront + cameraRight * u + cameraUp * v).normalize();
    Ray ray(cameraPos, rayDir);
    
    // Use full ray tracing
    Vec3 color = traceRay(ray, scene, &randState);
    
    // Convert to 0-255 range and apply gamma correction
    color.x = sqrtf(fminf(fmaxf(color.x, 0.0f), 1.0f));
    color.y = sqrtf(fminf(fmaxf(color.y, 0.0f), 1.0f));
    color.z = sqrtf(fminf(fmaxf(color.z, 0.0f), 1.0f));
    
    output[idx + 0] = (unsigned char)(color.x * 255.0f);
    output[idx + 1] = (unsigned char)(color.y * 255.0f);
    output[idx + 2] = (unsigned char)(color.z * 255.0f);
}

extern "C" void launchRayTracingKernel(unsigned char* output, int width, int height,
                                      const SceneData* scene, Vec3 cameraPos, 
                                      Vec3 cameraFront, Vec3 cameraUp, Vec3 cameraRight,
                                      float fov, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    rayTracingKernel<<<gridSize, blockSize, 0, stream>>>(
        output, width, height, scene, cameraPos, cameraFront, cameraUp, cameraRight, fov
    );
    
    cudaDeviceSynchronize();
}