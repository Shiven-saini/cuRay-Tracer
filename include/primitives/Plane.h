#ifndef PLANE_H
#define PLANE_H

#include "../math/Vec3.h"
#include "../math/Ray.h"
#include "Material.h"

struct Plane {
    Vec3 point;
    Vec3 normal;
    Material material;

    __host__ __device__ Plane() : normal(0, 1, 0) {}
    __host__ __device__ Plane(Vec3 point, Vec3 normal, Material material)
        : point(point), normal(normal.normalize()), material(material) {}

    __host__ __device__ bool intersect(const Ray& ray, float& t, Vec3& hitNormal) const {
        float denom = normal.dot(ray.direction);
        if (fabsf(denom) < 0.0001f) return false;

        t = (point - ray.origin).dot(normal) / denom;
        if (t <= 0.001f) return false;

        hitNormal = normal;
        return true;
    }
};

#endif