#include "intersect.h"

#include <cmath>

// Local helpers
static inline float dot3(const vec3 &a, const vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline float length3(const vec3 &v) {
    return std::sqrt(dot3(v, v));
}

static inline vec3 normalize3(const vec3 &v) {
    float len = length3(v);
    if (len <= 0.0f) return v;
    return vec3(v.x / len, v.y / len, v.z / len);
}

bool intersectSphere(const Sphere &s,
                     const Ray   &ray,
                     float t_min,
                     float t_max,
                     float &t_hit) {
    vec3 oc = ray.origin - s.center;
    float a = dot3(ray.dir, ray.dir);
    float b = 2.0f * dot3(oc, ray.dir);
    float c = dot3(oc, oc) - s.radius * s.radius;
    float disc = b*b - 4.0f*a*c;
    if (disc < 0.0f) return false;

    float sqrtD = std::sqrt(disc);
    float t0 = (-b - sqrtD) / (2.0f * a);
    float t1 = (-b + sqrtD) / (2.0f * a);

    float t = t0;
    if (t < t_min || t > t_max) {
        t = t1;
        if (t < t_min || t > t_max) return false;
    }

    t_hit = t;
    return true;
}
