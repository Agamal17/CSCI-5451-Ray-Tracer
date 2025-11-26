#pragma once

#include "scene.h"
#include "ray.h"

// Ray-sphere intersection.
// Returns true if there is a hit in [t_min, t_max] and writes the hit distance to t_hit.
bool intersectSphere(const Sphere &s,
                     const Ray   &ray,
                     float t_min,
                     float t_max,
                     float &t_hit);
