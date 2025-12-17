#pragma once
#include <limits>
#include <cuda_runtime.h> // For __host__ __device__
#include "types.h"

// ----------------- Material -----------------
struct Material {
    Color ambient;   // ar, ag, ab
    Color diffuse;   // dr, dg, db
    Color specular;  // sr, sg, sb
    double  ns;      // phong exponent
    Color trans;     // tr, tg, tb
    double  ior;     // index of refraction
};

struct HitInfo {
    double distance;
    Point3 point;
    Direction3 normal;
    Material* material;

    __host__ __device__ HitInfo() : distance(INFINITY), material(nullptr) {}
    __host__ __device__ HitInfo(double distance, Point3 point, Direction3 normal, Material* material)
        : distance(distance), point(point), normal(normal), material(material) {}
};

// Note: Primitive base class removed to avoid vtables on GPU.
// Sphere and Triangle are now standalone structs.

// ----------------- Sphere -----------------
struct Sphere {
    Point3    center;
    double    radius;
    Material* material;   // Pointer to material in Device memory

    __host__ __device__ Material* getMaterial() const { return material; }
    
    __device__ Direction3 get_normal_at_point(const Point3 &p) const;
};

// ----------------- Triangle -----------------
struct Triangle {
    Point3 v1, v2, v3;
    Direction3 n1, n2, n3;
    Direction3 triPlane; // Normal of the plane
    bool flat = true;    // true -> flat triangle else false
    Material* material;  // Pointer to material in Device memory

    __host__ __device__ Material* getMaterial() const { return material; }
    
    __device__ Direction3 get_normal_at_point(const Point3 &p) const;
};
