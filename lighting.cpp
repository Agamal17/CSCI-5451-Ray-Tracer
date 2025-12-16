#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include "Include/scene.h"
#include "Include/intersect.h"
#include "Include/lighting.h"
#include "Include/rayTrace.h"

static constexpr double EPS = 1e-4;

Color DirectionalLight::getContribution(
    const Scene& scene,
    const Ray& ray,
    HitInfo& hit)
{
    Color final_color(0, 0, 0);

    Direction3 N = hit.normal.normalized();
    Direction3 L = (-direction).normalized();   // surface → light
    Direction3 V = (-ray.dir).normalized();     // surface → camera

    Point3 p = hit.point + N * EPS;

    Ray shadowRay(p, L);
    HitInfo shadowHit;

    if (!FindIntersection(scene, shadowRay, shadowHit)) {
        // Diffuse
        double NdotL = std::max(0.0, dot(N, L));
        final_color += hit.material->diffuse * color * NdotL;

        // Specular (Blinn–Phong)
        Direction3 H = (V + L).normalized();
        double NdotH = std::max(0.0, dot(N, H));
        final_color += hit.material->specular * color *
                       pow(NdotH, hit.material->ns);
    }

    return final_color;
}

Color PointLight::getContribution(
    const Scene& scene,
    const Ray& ray,
    HitInfo& hit)
{
    Color final_color(0, 0, 0);

    Direction3 N = hit.normal.normalized();
    Direction3 V = (-ray.dir).normalized();

    Point3 p = hit.point + N * EPS;

    Direction3 toLight = position - p;
	double light_distance = toLight.length();
    Direction3 L = toLight.normalized();    // surface → light

    Ray shadowRay(p, L);
    HitInfo shadowHit;

    if (!(FindIntersection(scene, shadowRay, shadowHit) &&
          shadowHit.distance < light_distance)) {

        Color attenuated_color = color / (light_distance * light_distance);

        // Diffuse
        double NdotL = std::max(0.0, dot(N, L));
        final_color += hit.material->diffuse * attenuated_color * NdotL;

        // Specular
        Direction3 H = (L + V).normalized();
        double NdotH = std::max(0.0, dot(N, H));
        final_color += hit.material->specular * attenuated_color *
                       pow(NdotH, hit.material->ns);
    }

    return final_color;
}

Color SpotLight::getContribution(
    const Scene& scene,
    const Ray& ray,
    HitInfo& hit)
{
    Color final_color(0, 0, 0);

    Direction3 N = hit.normal.normalized();
    Direction3 V = (-ray.dir).normalized();

    Point3 p = hit.point + N * EPS;

    Direction3 toLight = position - p;
    double light_distance = toLight.length();
    Direction3 L = toLight.normalized();

    Ray shadowRay(p, L);
    HitInfo shadowHit;

    if (FindIntersection(scene, shadowRay, shadowHit) &&
        shadowHit.distance < light_distance)
        return final_color;

    // Angle between spotlight direction and hit direction
    double hitAngle =
        acos(dot((-toLight).normalized(), direction.normalized())) * 180.0 / M_PI;

    if (hitAngle > angle2)
        return final_color;

    double falloff = 1.0;
    if (hitAngle > angle1) {
        double t = (hitAngle - angle1) / (angle2 - angle1);
        falloff = std::max(0.0, 1.0 - t);
    }

    Color attenuated_color =
        (color / (light_distance * light_distance)) * falloff;

    // Diffuse
    double NdotL = std::max(0.0, dot(N, L));
    final_color += hit.material->diffuse * attenuated_color * NdotL;

    // Specular
    Direction3 H = (L + V).normalized();
    double NdotH = std::max(0.0, dot(N, H));
    final_color += hit.material->specular * attenuated_color *
                   pow(NdotH, hit.material->ns);

    return final_color;
}

Color ApplyLighting(
    const Scene& scene,
    Ray& ray,
    HitInfo& hit,
    int depth)
{
    Color color = hit.material->ambient * scene.ambient_light;

    for (Light* light : scene.lights) {
        color += light->getContribution(scene, ray, hit);
    }

    if (depth > 0) {
        // Refraction
        Ray refraction = Refract(ray, hit);
        color += hit.material->trans *
                 rayTrace(refraction, depth - 1, scene);

        // Reflection
        Ray reflection = Reflect(ray, hit);
        color += hit.material->specular *
                 rayTrace(reflection, depth - 1, scene);
    }

    return color;
}
