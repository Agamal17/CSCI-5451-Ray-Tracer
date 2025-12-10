#include "Include/primitive.h"

Direction3 Triangle::get_normal_at_point(const Point3 &p) const
{
    if (flat) return n1;

    vec3 e0 = v2 - v1;
    vec3 e1 = v3 - v1;
    vec3 vp = p  - v1;

    float d00 = dot(e0, e0);
    float d01 = dot(e0, e1);
    float d11 = dot(e1, e1);
    float d20 = dot(vp, e0);
    float d21 = dot(vp, e1);

    float denom = d00 * d11 - d01 * d01;
    float b2 = (d11 * d20 - d01 * d21) / denom;
    float b3 = (d00 * d21 - d01 * d20) / denom;
    float b1 = 1.0f - b2 - b3;

    vec3 n = (b1 * n1 + b2 * n2 + b3 * n3).normalized();
    return Direction3(n);
}

Direction3 Sphere::get_normal_at_point(const Point3 &p) const {
    vec3 n = p - center;

    float len2 = n.x * n.x + n.y * n.y + n.z * n.z;
    if (len2 > 0.0f) {
        float invLen = 1.0f / std::sqrt(len2);
        n.x *= invLen;
        n.y *= invLen;
        n.z *= invLen;
    }

    return Direction3(n);
}

Material* Sphere::getMaterial() const {
    return material;
}

Material* Triangle::getMaterial() const {
    return material;
}
