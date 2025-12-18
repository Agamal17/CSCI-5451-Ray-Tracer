#include "Include/intersect.cuh"
#include <cmath>
#include <cfloat>
#include <limits>

// ----------------------------------------------------------------------------
// [BVH] Intersection Helper
// ----------------------------------------------------------------------------
__device__ bool intersectAABB(const Ray& r, const AABB& box, float& t_in) {
    float tx1 = (box.min.x - r.origin.x) * r.inv_dir.x;
    float tx2 = (box.max.x - r.origin.x) * r.inv_dir.x;
    float tmin = fminf(tx1, tx2);
    float tmax = fmaxf(tx1, tx2);

    float ty1 = (box.min.y - r.origin.y) * r.inv_dir.y;
    float ty2 = (box.max.y - r.origin.y) * r.inv_dir.y;
    tmin = fmaxf(tmin, fminf(ty1, ty2));
    tmax = fminf(tmax, fmaxf(ty1, ty2));

    float tz1 = (box.min.z - r.origin.z) * r.inv_dir.z;
    float tz2 = (box.max.z - r.origin.z) * r.inv_dir.z;
    tmin = fmaxf(tmin, fminf(tz1, tz2));
    tmax = fminf(tmax, fmaxf(tz1, tz2));

    if (tmax >= tmin && tmax > 0.0f) {
        t_in = tmin;
        return true;
    }
    return false;
}

// ----------------------------------------------------------------------------
// Primitive Intersections (Adapted to device)
// ----------------------------------------------------------------------------

__device__ float intersectSphere(const Ray ray, const Sphere s) {
    // Optimized for float
    Direction3 oc = ray.origin - s.center;
    float a = dot(ray.dir, ray.dir);
    float b = 2.0f * dot(oc, ray.dir);
    float c = dot(oc, oc) - s.radius * s.radius;
    float disc = b*b - 4.0f*a*c;
    
    if (disc < 0.0f) return 1e30f; // No hit

    float sqrtD = sqrtf(disc);
    float t0 = (-b - sqrtD) / (2.0f * a);
    float t1 = (-b + sqrtD) / (2.0f * a);

    float t = t0;
    if (t < 1e-4f) {
        t = t1;
        if (t < 1e-4f) return 1e30f;
    }
    return t;
}

__device__ float rayTriangleIntersect(const Ray ray, const Triangle tri) {
    Direction3 n = tri.triPlane;
    float denom = dot(n, ray.dir);
    if (fabsf(denom) < 1e-7f) return 1e30f;

    float t = dot(n, tri.v1 - ray.origin) / denom;
    if (t < 1e-4f) return 1e30f;

    Point3 hitPoint = ray.origin + ray.dir * t;

    Direction3 c1 = cross(tri.v2 - tri.v1, hitPoint - tri.v1);
    Direction3 c2 = cross(tri.v3 - tri.v2, hitPoint - tri.v2);
    Direction3 c3 = cross(tri.v1 - tri.v3, hitPoint - tri.v3);

    float d1 = dot(c1, n);
    float d2 = dot(c2, n);
    float d3 = dot(c3, n);

    if ((d1 >= 0 && d2 >= 0 && d3 >= 0) || (d1 <= 0 && d2 <= 0 && d3 <= 0)) {
        return t;
    }
    return 1e30f;
}

// ----------------------------------------------------------------------------
// [BVH] Traversal
// ----------------------------------------------------------------------------

template <typename TPrimitive, typename TFunc>
__device__ void traverseBVH(
    const BVHNode* nodes, 
    const TPrimitive* primitives, 
    const Ray& ray, 
    HitInfo* closestHit, 
    float& closest_t,
    TFunc intersectFunc) 
{
    if (nodes == nullptr) return;

    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0; // Push root

    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        const BVHNode& node = nodes[nodeIdx];

        float t_box;
        if (!intersectAABB(ray, node.bounds, t_box)) {
            continue;
        }
        if (t_box > closest_t) {
            continue; // Optimization
        }

        if (node.count > 0) { 
            // LEAF
            for (int i = 0; i < node.count; i++) {
                int primIdx = node.leftFirst + i;
                float t = intersectFunc(ray, primitives[primIdx]);
                if (t < closest_t && t > 1e-4f) {
                    closest_t = t;
                    closestHit->distance = t;
                    closestHit->point = ray.origin + ray.dir * t;
                    // Note: We access material directly from primitive
                    closestHit->material_index = primitives[primIdx].material_index;
                    
                    // Normal calculation requires polymorphism or switch.
                    // For now, re-use primitive specific normal logic logic
                    // This part is slightly generic, assuming TPrimitive has get_normal logic inline 
                    // or we handle it here. 
                    // To keep it simple, we don't compute normal here. We compute it AFTER the loop 
                    // or inside specific lambdas if needed.
                    // However, we MUST store which primitive was hit to compute normal later
                }
            }
        } else {
            // INTERNAL
            // Push children (optimization: order by distance could be done here)
            stack[stackPtr++] = node.leftFirst + 1; // Right
            stack[stackPtr++] = node.leftFirst;     // Left
        }
    }
}

// ----------------------------------------------------------------------------
// Main Intersection Entry Point
// ----------------------------------------------------------------------------

__device__ bool FindIntersection(const DeviceScene* scene, const Ray ray, HitInfo* hit) {
    float closest_t = 1e30f;
    hit->distance = 1e30f;
    const void* hitPrimitive = nullptr; // Track what we hit for normal calc
    int hitType = 0; // 0=None, 1=Sphere, 2=Triangle

    // 1. Traverse Spheres
    if (scene->d_sphereBVH) {
         traverseBVH(scene->d_sphereBVH, scene->d_spheres, ray, hit, closest_t, 
            [&](const Ray& r, const Sphere& s) { return intersectSphere(r, s); });
         
         // If we updated distance, we hit a sphere
         if (hit->distance < 1e29f && hitType == 0) hitType = 1; // Simplification, need better tracking
    } else {
        // Fallback for linear if BVH fails/empty (though BVH covers all)
    }
    
    // To correctly track normals, traverseBVH needs to populate the normal. 
    // Since traverseBVH is generic, let's specialize slightly or pass a lambda that updates normal.
    // Re-implementation of traversal calls with specific normal updates:
    
    // -- SPHERES --
    if (scene->d_sphereBVH) {
        int stack[64]; int sp = 0; stack[sp++] = 0;
        while(sp > 0) {
            int idx = stack[--sp];
            const BVHNode& node = scene->d_sphereBVH[idx];
            float t_box;
            if(!intersectAABB(ray, node.bounds, t_box) || t_box > closest_t) continue;
            
            if(node.count > 0) {
                for(int i=0; i<node.count; i++) {
                    const Sphere& s = scene->d_spheres[node.leftFirst + i];
                    float t = intersectSphere(ray, s);
                    if(t < closest_t && t > 1e-4f) {
                        closest_t = t;
                        hit->distance = t;
                        hit->point = ray.origin + ray.dir * t;
                        hit->material_index = s.material_index;
                        hit->normal = (hit->point - s.center).normalized();
                        if(dot(hit->normal, ray.dir) > 0) hit->normal = -hit->normal;
                    }
                }
            } else {
                stack[sp++] = node.leftFirst + 1;
                stack[sp++] = node.leftFirst;
            }
        }
    }

    // -- TRIANGLES --
    if (scene->d_triangleBVH) {
        int stack[64]; int sp = 0; stack[sp++] = 0;
        while(sp > 0) {
            int idx = stack[--sp];
            const BVHNode& node = scene->d_triangleBVH[idx];
            float t_box;
            if(!intersectAABB(ray, node.bounds, t_box) || t_box > closest_t) continue;
            
            if(node.count > 0) {
                for(int i=0; i<node.count; i++) {
                    const Triangle& tri = scene->d_triangles[node.leftFirst + i];
                    float t = rayTriangleIntersect(ray, tri);
                    if(t < closest_t && t > 1e-4f) {
                        closest_t = t;
                        hit->distance = t;
                        hit->point = ray.origin + ray.dir * t;
                        hit->material_index = tri.material_index;
                        
                        // Normal Calc
                        if (tri.flat) {
                            hit->normal = tri.n1;
                        } else {
                            // Interpolate normal using barycentric coords
                            Direction3 e0 = tri.v2 - tri.v1;
                            Direction3 e1 = tri.v3 - tri.v1;
                            Direction3 vp = hit->point - tri.v1;
                            float d00 = dot(e0, e0);
                            float d01 = dot(e0, e1);
                            float d11 = dot(e1, e1);
                            float d20 = dot(vp, e0);
                            float d21 = dot(vp, e1);
                            float denom = d00 * d11 - d01 * d01;
                            float v = (d11 * d20 - d01 * d21) / denom;
                            float w = (d00 * d21 - d01 * d20) / denom;
                            float u = 1.0f - v - w;
                            hit->normal = (u * tri.n1 + v * tri.n2 + w * tri.n3).normalized();
                        }
                        if(dot(hit->normal, ray.dir) > 0) hit->normal = -hit->normal;
                    }
                }
            } else {
                stack[sp++] = node.leftFirst + 1;
                stack[sp++] = node.leftFirst;
            }
        }
    }

    return (closest_t < 1e29f);
}
