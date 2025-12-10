#pragma once
#include "types.h"
#include "ray.h"

struct Scene;

struct Light {
    Color color;

    Light(Color color): color(color) {}
    virtual Color getContribution(const Scene& scene, const Ray& ray, HitInfo& hit)  = 0;
};

struct DirectionalLight: public Light{
    Direction3 direction;

    DirectionalLight(Color color, Direction3 direction): Light(color), direction(direction) {}
    Color getContribution(const Scene& scene, const Ray& ray, HitInfo& hit);
};

struct PointLight: public Light{
    Point3 position;

    PointLight(Color color, Point3 position): Light(color), position(position) {}
    Color getContribution(const Scene& scene, const Ray& ray, HitInfo& hit);
};

struct SpotLight: public Light{
    Point3 position;
    Direction3 direction;
    double angle1;
    double angle2;

    SpotLight(Color color, Point3 position, Direction3 direction, double angle1, double angle2): Light(color), position(position), direction(direction), angle1(angle1), angle2(angle2) {}
    Color getContribution(const Scene& scene, const Ray& ray, HitInfo& hit);
};

Color ApplyLighting(const Scene& scene,
                    const Ray &ray,
                    HitInfo &hit,
                    int depth);