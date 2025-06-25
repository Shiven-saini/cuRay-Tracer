#ifndef MATERIAL_H
#define MATERIAL_H

#include "../math/Vec3.h"

enum MaterialType {
    DIFFUSE,
    REFLECTIVE,
    REFRACTIVE,
    EMISSIVE
};

struct Material {
    Vec3 color;
    MaterialType type;
    float roughness;
    float refractiveIndex;
    float reflectivity;

    __host__ __device__ Material() 
        : color(0.8f, 0.8f, 0.8f), type(DIFFUSE), roughness(1.0f), 
          refractiveIndex(1.0f), reflectivity(0.0f) {}

    __host__ __device__ Material(Vec3 color, MaterialType type, float roughness = 1.0f, 
                                 float refractiveIndex = 1.0f, float reflectivity = 0.0f)
        : color(color), type(type), roughness(roughness), 
          refractiveIndex(refractiveIndex), reflectivity(reflectivity) {}
};

#endif