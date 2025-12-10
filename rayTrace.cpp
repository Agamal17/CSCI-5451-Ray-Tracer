#include "Include/intersect.h"
#include "Include/lighting.h"
#include "Include/rayTrace.h"


Color rayTrace(const Ray &ray, const int max_depth, const Scene& scene) {
    // Base Case: Stop the recursion if max depth is reached
    if (max_depth <= 0) {
        return Color(0, 0, 0);
    }

    bool b_hit = false;
    HitInfo hit;

    b_hit = FindIntersection(scene, ray, hit);
    if (b_hit) {
        return ApplyLighting(scene, ray, hit, max_depth);
    }

    return scene.background;
}

Ray Reflect(const Ray &ray, const HitInfo& hit){
    // TODO: To be done by Neiil
    return Ray();
}

Ray Refract(const Ray &ray, const HitInfo& hit){
    // TODO: To be done by Neiil
    return Ray();
}