#pragma once
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cuda_runtime.h> // Required for __host__ __device__ macros

// Removed "using std::..." to avoid conflicts with CUDA math functions on device

// Small vector library
// Represents a vector as 3 doubles
struct vec3 {
    double x, y, z;

    __host__ __device__ vec3(double x, double y, double z) : x(x), y(y), z(z) {}
    __host__ __device__ vec3() : x(0), y(0), z(0) {}

    __host__ __device__ vec3 operator-() const {
        return vec3(-x, -y, -z);
    }

    // Clamp each component (used to clamp pixel colors)
    __host__ __device__ vec3 clampTo1() {
        // Use global fmin (CUDA compatible) instead of std::fmin
        return vec3(fmin(x, 1.0), fmin(y, 1.0), fmin(z, 1.0));
    }

    // Compute vector length
    __host__ __device__ double length() {
        return sqrt(x * x + y * y + z * z);
    }

    // Create a unit-length vector
    __host__ __device__ vec3 normalized() const {
        double len = sqrt(x * x + y * y + z * z);
        // Avoid division by zero if necessary, though original code didn't check
        if (len == 0.0) return vec3(0, 0, 0);
        return vec3(x / len, y / len, z / len);
    }
};

// Multiply double and vector
__host__ __device__ inline vec3 operator*(double f, vec3 a) {
    return vec3(a.x * f, a.y * f, a.z * f);
}

// Vector-vector dot product
__host__ __device__ inline double dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Vector-vector cross product
__host__ __device__ inline vec3 cross(vec3 a, vec3 b) {
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// Vector addition
__host__ __device__ inline vec3 operator+(vec3 a, vec3 b) {
    return vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// Vector subtraction
__host__ __device__ inline vec3 operator-(vec3 a, vec3 b) {
    return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// Useful for optimization (avoids expensive sqrt)
__host__ __device__ inline double length_squared(const vec3 v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

// Allow vec * double
__host__ __device__ inline vec3 operator*(const vec3& a, double f) {
    return vec3(a.x * f, a.y * f, a.z * f);
}

// Element-wise (Hadamard) multiplication
__host__ __device__ inline vec3 operator*(const vec3 a, const vec3 b) {
    return vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

// operator<< uses std::ostream, which is not available on the device.
// We mark this as __host__ only.
__host__ inline std::ostream& operator<<(std::ostream& os, const vec3 v) {
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
}
