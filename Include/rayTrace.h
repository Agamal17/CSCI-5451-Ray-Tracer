#pragma once
#include "types.h"
#include "ray.h"
#include "scene.h"

Color rayTrace(Ray &ray, const int max_depth, const Scene& scene);
Ray Reflect(Ray &ray, HitInfo& hit);
Ray Refract(Ray &ray, HitInfo& hit);