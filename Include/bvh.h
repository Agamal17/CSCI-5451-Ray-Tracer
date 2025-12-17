#pragma once
#include "vec3.h"
#include <algorithm>
#include <cuda_runtime.h>

struct AABB {
    Point3 min;
    Point3 max;

    __host__ __device__ AABB() : min(1e30, 1e30, 1e30), max(-1e30, -1e30, -1e30) {}
    __host__ __device__ AABB(const Point3& min, const Point3& max) : min(min), max(max) {}

    __host__ __device__ void expand(const Point3& p) {
        min = Point3(fmin(min.x, p.x), fmin(min.y, p.y), fmin(min.z, p.z));
        max = Point3(fmax(max.x, p.x), fmax(max.y, p.y), fmax(max.z, p.z));
    }

    __host__ __device__ void expand(const AABB& box) {
        min = Point3(fmin(min.x, box.min.x), fmin(min.y, box.min.y), fmin(min.z, box.min.z));
        max = Point3(fmax(max.x, box.max.x), fmax(max.y, box.max.y), fmax(max.z, box.max.z));
    }
    
    __host__ __device__ Point3 centroid() const {
        return min + (max - min) * 0.5;
    }
};

struct BVHNode {
    AABB bounds;
    int leftFirst;  // If leaf: index into primitive array. If internal: index of left child.
    int count;      // If leaf: number of primitives. If internal: 0.
};
