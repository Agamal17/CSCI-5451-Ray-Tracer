#pragma once
#include <cmath>
#include "types.cuh"

// ----------------- Material -----------------
// Simple data structure. Safe for device code.
struct Material {
    Color ambient;
    Color diffuse;
    Color specular;
    double  ns;
    Color trans;
    double  ior;
};

// In your shared header (e.g., primitives.cuh)
enum PrimitiveType {
    PRIMITIVE_TYPE_NONE = 0,
    PRIMITIVE_TYPE_SPHERE,
    PRIMITIVE_TYPE_TRIANGLE
    // Add other types as needed
};

// ----------------- HitInfo -----------------
struct HitInfo {
    double distance;
    Point3 point;
    Direction3 normal;
    int material_index;

    // --- NEW FIELDS ---
    PrimitiveType primitive_type; // Tells us what kind of object was hit
    int primitive_index;          // Index into the d_spheres or d_triangles array
};

// ----------------- Sphere -----------------
struct Sphere {
    Point3    center;
    double    radius;
    int       material_index;

    __device__ Direction3 get_normal_at_point(const Point3 p) const;
};

// ----------------- Triangle -----------------
struct Triangle {
    Point3 v1, v2, v3;
    Direction3 n1, n2, n3;
    Direction3 triPlane;
    bool flat = true;
    int     material_index;

    __device__ Direction3 get_normal_at_point(const Point3 p) const;
};