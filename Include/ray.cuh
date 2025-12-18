#pragma once
#include "types.cuh"

struct Ray {
    Point3     origin;
    Direction3 dir;
    Direction3 inv_dir; // [BVH] Precomputed inverse direction

    __host__ __device__ Ray(Point3 origin, Direction3 dir)
        : origin(origin), dir(dir.normalized()) {
        
        // [BVH] Precompute inverse direction for AABB slabs method
        // Avoid division by zero by using a small epsilon if needed, 
        // though standard IEEE 754 float division handles infinity correctly for slab tests.
        inv_dir = Direction3(1.0f / this->dir.x, 1.0f / this->dir.y, 1.0f / this->dir.z);
    }

    __host__ __device__ Ray() = default;
};
