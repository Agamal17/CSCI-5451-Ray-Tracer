#pragma once
#include "types.cuh"
#include "ray.cuh"
#include "primitive.cuh"
#include "scene.cuh"

// Ray-sphere intersection.
// Returns distance to intersection or infinity if no hit.
__device__ double intersectSphere(const Ray ray, const Sphere s);

// Ray-triangle intersection
__device__ double rayTriangleIntersect(const Ray ray, const Triangle triangle);

__device__ bool FindIntersection(const DeviceScene* scene, const Ray ray, HitInfo* hit);
