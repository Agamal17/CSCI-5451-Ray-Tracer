#pragma once
#include "types.cuh"
#include "ray.cuh"
#include "scene.cuh"

__device__ Color rayTrace(const DeviceScene* scene, Ray ray, const int max_depth);
__device__ Ray Reflect(Ray ray, HitInfo hit);
__device__ Ray Refract(const DeviceScene* scene, Ray ray, HitInfo hit);