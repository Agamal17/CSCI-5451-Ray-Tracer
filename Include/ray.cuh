#pragma once
#include "types.cuh"

struct Ray {
    Point3     origin;
    Direction3 dir;

    // 1. Host/Device Qualified Constructor: Must be callable from both host and device.
    // 2. The dir.normalized() method MUST also be __host__ __device__ qualified.
    __host__ __device__ Ray(Point3 origin, Direction3 dir)
        : origin(origin), dir(dir.normalized()) {}

    // Default constructor is fine, but mark it for clarity.
    __host__ __device__ Ray() = default;
};
