#include "Include/intersect.cuh"
#include <cmath>
#include <cfloat>

static constexpr double T_MIN = 0.0001; // Epsilon

__device__ double intersectSphere(const Ray ray, const Sphere s) {
    const double INF = INFINITY;
    
    Direction3 oc = ray.origin - s.center;
    double a = dot(ray.dir, ray.dir);
    double b = 2.0 * dot(oc, ray.dir);
    double c = dot(oc, oc) - s.radius * s.radius;
    double disc = b*b - 4.0*a*c;
    
    if (disc < 0.0) return INF;

    double sqrtD = std::sqrt(disc);
    double t0 = (-b - sqrtD) / (2.0 * a);
    double t1 = (-b + sqrtD) / (2.0 * a);


    double t = t0;
    if (t < 1e-4) {
        t = t1;
        if (t < 1e-4) return INF;
    }

    return t;
}

__device__ double rayTriangleIntersect(const Ray ray, const Triangle triangle)
{
    const double INF = INFINITY;

    Direction3 n = triangle.triPlane;
    double denom = dot(n, ray.dir);
    if (std::abs(denom) < 1e-7) return INF;

    double t = dot(n, triangle.v1 - ray.origin) / denom;
    if (t < 1e-4) return INF;

    Point3 hitPoint = ray.origin + ray.dir * t;

    Direction3 c1 = cross(triangle.v2 - triangle.v1, hitPoint - triangle.v1);
    Direction3 c2 = cross(triangle.v3 - triangle.v2, hitPoint - triangle.v2);
    Direction3 c3 = cross(triangle.v1 - triangle.v3, hitPoint - triangle.v3);

    double d1 = dot(c1, n);
    double d2 = dot(c2, n);
    double d3 = dot(c3, n);

    if ( (d1 >= 0 && d2 >= 0 && d3 >= 0) ||
         (d1 <= 0 && d2 <= 0 && d3 <= 0) )
    {
        return t;
    }

    return INF;
}


__device__ bool FindIntersection(const DeviceScene* d_scene, const Ray ray, HitInfo* hit) {
    float closest_t = FLT_MAX;

    // Reset hit structure to track the closest intersection
    hit->distance = FLT_MAX;
    hit->primitive_type = PRIMITIVE_TYPE_NONE;
    hit->primitive_index = -1;

    // Temporary variables for intersection test result
    double t_d;

    // --- 1. Check Spheres (C-style iteration over device array) ---
    for (int i = 0; i < d_scene->num_spheres; ++i) {
        // Access the sphere using array indexing
        const Sphere& sphere = d_scene->d_spheres[i];

        // intersectSphere must be a __device__ function
        t_d = intersectSphere(ray, sphere);

        // rayTriangleIntersect returns infinity on miss
        if (t_d != INFINITY) {
            if (t_d > T_MIN && t_d < closest_t) {
                closest_t = t_d;

                // CRITICAL: Update HitInfo with type and index
                hit->primitive_type = PRIMITIVE_TYPE_SPHERE;
                hit->primitive_index = i;
                hit->material_index = sphere.material_index; // Store material index
            }
        }
    }

    // --- 2. Check Triangles (C-style iteration over device array) ---
    for (int i = 0; i < d_scene->num_triangles; ++i) {
        // Access the triangle using array indexing
        const Triangle& tri = d_scene->d_triangles[i];

        // rayTriangleIntersect must be a __device__ function
        t_d = rayTriangleIntersect(ray, tri);

        if (t_d != INFINITY) {
            if (t_d > T_MIN && t_d < closest_t) {
                closest_t = t_d;

                // CRITICAL: Update HitInfo with type and index
                hit->primitive_type = PRIMITIVE_TYPE_TRIANGLE;
                hit->primitive_index = i;
                hit->material_index = tri.material_index; // Store material index
            }
        }
    }

    // --- 3. Populate HitInfo if we hit something ---
    if (hit->primitive_type != PRIMITIVE_TYPE_NONE) {
        hit->distance = closest_t;
        hit->point = ray.origin + ray.dir * closest_t;

        Direction3 surface_normal;

        // CRITICAL: Use switch statement to replace polymorphism and calculate the normal
        switch (hit->primitive_type) {
            case PRIMITIVE_TYPE_SPHERE: {
                const Sphere& s = d_scene->d_spheres[hit->primitive_index];
                // The normal calculation function must be __device__
                surface_normal = s.get_normal_at_point(hit->point);
                break;
            }
            case PRIMITIVE_TYPE_TRIANGLE: {
                const Triangle& t = d_scene->d_triangles[hit->primitive_index];
                // The normal calculation function must be __device__
                surface_normal = t.get_normal_at_point(hit->point);
                break;
            }
            default:
                // Should not happen if primitive_type != NONE
                surface_normal = Direction3(0, 0, 0);
                break;
        }

        hit->normal = surface_normal.normalized();

        // Ensure normal is opposite to viewing ray (for front faces)
        if (dot(hit->normal, ray.dir) > 0) {
            hit->normal = -hit->normal;
        }

        // Material is already stored as an index in hit->material_index
        return true;
    }

    return false;
}
