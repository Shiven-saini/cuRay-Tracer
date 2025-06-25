#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }

    __host__ __device__ Vec3 operator-(const Vec3& v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }

    __host__ __device__ Vec3 operator*(float t) const {
        return Vec3(x * t, y * t, z * t);
    }

    __host__ __device__ Vec3 operator*(const Vec3& v) const {
        return Vec3(x * v.x, y * v.y, z * v.z);
    }

    __host__ __device__ float dot(const Vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    __host__ __device__ Vec3 cross(const Vec3& v) const {
        return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }

    __host__ __device__ float length() const {
        return sqrtf(x * x + y * y + z * z);
    }

    __host__ __device__ float lengthSquared() const {
        return x * x + y * y + z * z;
    }

    __host__ __device__ Vec3 normalize() const {
        float len = length();
        return len > 0 ? *this * (1.0f / len) : Vec3(0, 0, 0);
    }

    __host__ __device__ Vec3 reflect(const Vec3& normal) const {
        return *this - normal * (2.0f * this->dot(normal));
    }

    __host__ __device__ Vec3 refract(const Vec3& normal, float eta) const {
        float cosI = -this->dot(normal);
        float sinT2 = eta * eta * (1.0f - cosI * cosI);
        if (sinT2 >= 1.0f) return Vec3(0, 0, 0); // Total internal reflection
        return *this * eta + normal * (eta * cosI - sqrtf(1.0f - sinT2));
    }
};

// Global operators for scalar multiplication
__host__ __device__ inline Vec3 operator*(float t, const Vec3& v) {
    return Vec3(v.x * t, v.y * t, v.z * t);
}

#endif