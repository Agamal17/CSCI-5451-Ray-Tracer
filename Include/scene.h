#pragma once

#include "types.h"
#include <vector>
#include <string>

struct Material {
    Color3 ambient;   // ar, ag, ab
    Color3 diffuse;   // dr, dg, db
    Color3 specular;  // sr, sg, sb
    float  ns;        // phong exponent
    Color3 trans;     // tr, tg, tb
    float  ior;       // index of refraction
};
struct Triangle {
    int  v[3];               // indices into Scene::vertices
    int  n[3];               // indices into Scene::normals (or -1 if none)
    int  material_id;        // which material to use
    bool has_vertex_normals; // true if n[] are valid
};
struct Sphere {
    Point3 center;
    float  radius;
    int    material_id;

    // NEW method to get normal at a point on the sphere surface
    Normal3 get_normal_at_point(const Point3 &p) const;
};

struct Scene {
    // camera
    Point3     camera_pos;
    Direction3 camera_fwd;
    Direction3 camera_up;
    float      camera_fov_ha;

    // global settings
    Color3 background;
    Color3 ambient_light;

    std::vector<Material> materials;
    std::vector<Sphere>   spheres;

    std::vector<Point3>   vertices;  
    std::vector<Normal3>  normals;   // same
    std::vector<Triangle> triangles;
};


// Parses the scene file, fills a Scene, and also sets output image size & name.
Scene parseSceneFile(const std::string &filename,
                     int &img_width,
                     int &img_height,
                     std::string &imgName);
