#pragma once

#include <vector>
#include <string>
#include "types.h"

// ----------------- Material -----------------
struct Material {
    Color3 ambient;   // ar, ag, ab
    Color3 diffuse;   // dr, dg, db
    Color3 specular;  // sr, sg, sb
    float  ns;        // phong exponent
    Color3 trans;     // tr, tg, tb
    float  ior;       // index of refraction
};

struct Primitive {
    virtual Direction3 get_normal_at_point(const Point3 &p) const = 0;
};

// ----------------- Sphere -----------------
struct Sphere : public Primitive {
    Point3    center;
    float     radius;
    int       material_id;   // index into Scene::materials

    // normal at a point on the surface
    Direction3 get_normal_at_point(const Point3 &p) const override;
};

// ----------------- Triangle -----------------
struct Triangle : public Primitive {
    Point3 v1, v2, v3;
    Direction3 n1, n2, n3;
    Direction3 triPlane;
    bool flat = true;  // true -> flat triangle else false
    int material_id;

    Direction3 get_normal_at_point(const Point3 &p) const override;
};

// ----------------- Scene -----------------
struct Scene {
    // camera
    Point3     camera_pos;
    Direction3 camera_fwd;
    Direction3 camera_up;
    float      camera_fov_ha;

    // global settings
    Color3 background;
    Color3 ambient_light;

    // materials & primitives
    std::vector<Material> materials;
    std::vector<Sphere>   spheres;

    // triangle data
    std::vector<Point3>     vertices;  // positions
    std::vector<Direction3> normals;   // per-vertex normals
    std::vector<Triangle>   triangles;
};

// parse scene file and fill scene + output image info
Scene parseSceneFile(const std::string &filename,
                     int &img_width,
                     int &img_height,
                     std::string &imgName);
