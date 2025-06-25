#ifndef RAYTRACING_KERNEL_H
#define RAYTRACING_KERNEL_H

#include "../Scene.h"
#include "../Camera.h"
#include <cuda_runtime.h>

extern "C" {
    void launchRayTracingKernel(float4* output, int width, int height,
                               const SceneData* scene, Vec3 cameraPos, 
                               Vec3 cameraFront, Vec3 cameraUp, Vec3 cameraRight,
                               float fov, cudaStream_t stream = 0);
}

#endif