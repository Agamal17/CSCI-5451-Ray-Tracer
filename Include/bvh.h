#pragma once
#include "vec3.h"
#include <algorithm>
#include <cuda_runtime.h>

struct AABB {
    Point3 min;
    Point3 max;

    __host__ __device__ AABB() 
        : min(1e30f, 1e30f, 1e30f), max(-1e30f, -1e30f, -1e30f) {}

    __host__ __device__ AABB(const Point3& min, const Point3& max) 
        : min(min), max(max) {}

    __host__ __device__ void expand(const Point3& p) {
        min.x = fminf(min.x, p.x);
        min.y = fminf(min.y, p.y);
        min.z = fminf(min.z, p.z);
        max.x = fmaxf(max.x, p.x);
        max.y = fmaxf(max.y, p.y);
        max.z = fmaxf(max.z, p.z);
    }

    __host__ __device__ void expand(const AABB& box) {
        min.x = fminf(min.x, box.min.x);
        min.y = fminf(min.y, box.min.y);
        min.z = fminf(min.z, box.min.z);
        max.x = fmaxf(max.x, box.max.x);
        max.y = fmaxf(max.y, box.max.y);
        max.z = fmaxf(max.z, box.max.z);
    }

    __host__ __device__ Point3 centroid() const {
        return (min + max) * 0.5f;
    }
};

struct BVHNode {
    AABB bounds;
    int leftFirst;  // Internal: Index of left child. Leaf: Index into primitive array.
    int count;      // Internal: 0. Leaf: Number of primitives.
};
