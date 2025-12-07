#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <cmath>

#include "Include/camera.h"


// Small helper functions local to this translation unit
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

static inline vec3 cross3(const vec3 &a, const vec3 &b) {
    return vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

Ray generateCameraRay(const Scene &scene,
                      int img_width,
                      int img_height,
                      int i,
                      int j) {
    vec3 fwd   = normalize3(scene.camera_fwd);
    vec3 right = normalize3(cross3(fwd, scene.camera_up));
    vec3 up    = normalize3(cross3(right, fwd)); // orthonormal up

    float aspect = float(img_width) / float(img_height);
    float ha_rad = scene.camera_fov_ha * float(M_PI) / 180.0f;
    float imgPlaneH = 2.0f * std::tan(ha_rad);   // image plane height
    float imgPlaneW = imgPlaneH * aspect;        // image plane width

    // Map pixel center to [-0.5, 0.5] in both x and y
    float u = ( (i + 0.5f) / float(img_width)  - 0.5f ) * imgPlaneW;
    float v = ( (j + 0.5f) / float(img_height) - 0.5f ) * imgPlaneH;

    // In image coordinates y increases downward, so subtract v * up
    vec3 dir = normalize3( fwd + u * right - v * up );

    Ray ray;
    ray.origin = scene.camera_pos;
    ray.dir    = dir;
    return ray;
}
