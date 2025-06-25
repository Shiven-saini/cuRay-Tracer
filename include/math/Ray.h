#ifndef RAY_H
#define RAY_H

#include "Vec3.h"

struct Ray {
    Vec3 origin;
    Vec3 direction;

    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Vec3& origin, const Vec3& direction) 
        : origin(origin), direction(direction) {}

    __host__ __device__ Vec3 at(float t) const {
        return origin + direction * t;
    }
};

#endif