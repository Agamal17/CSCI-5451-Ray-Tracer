#pragma once
#include "types.h"
#include "ray.h"
#include "scene.h"

Color rayTrace(const Ray &ray, const int max_depth, const Scene& scene);
Ray Reflect(const Ray &ray, const HitInfo& hit);
Ray Refract(const Ray &ray, const HitInfo& hit);