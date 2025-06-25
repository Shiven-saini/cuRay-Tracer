#ifndef SPHERE_H
#define SPHERE_H

#include "../math/Vec3.h"
#include "../math/Ray.h"
#include "Material.h"

struct Sphere {
    Vec3 center;
    float radius;
    Material material;

    __host__ __device__ Sphere() : radius(1.0f) {}
    __host__ __device__ Sphere(Vec3 center, float radius, Material material)
        : center(center), radius(radius), material(material) {}

    __host__ __device__ bool intersect(const Ray& ray, float& t, Vec3& normal) const {
        Vec3 oc = ray.origin - center;
        float a = ray.direction.dot(ray.direction);
        float b = 2.0f * oc.dot(ray.direction);
        float c = oc.dot(oc) - radius * radius;
        float discriminant = b * b - 4 * a * c;

        if (discriminant < 0) return false;

        float t1 = (-b - sqrtf(discriminant)) / (2.0f * a);
        float t2 = (-b + sqrtf(discriminant)) / (2.0f * a);

        t = (t1 > 0.001f) ? t1 : t2;
        if (t <= 0.001f) return false;

        Vec3 hitPoint = ray.at(t);
        normal = (hitPoint - center).normalize();
        return true;
    }
};

#endif