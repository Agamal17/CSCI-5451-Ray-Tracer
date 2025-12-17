#pragma once
#include <vector>
#include <string>
#include "types.cuh"
#include "primitive.cuh"
#include "lighting.cuh"
#include "bvh.cuh" // [BVH] Added include

// ----------------- Scene -----------------
struct HostSceneTemp {
    std::vector<DeviceLight> h_lights;
    std::vector<Sphere> h_spheres;
    std::vector<Triangle> h_triangles;
    std::vector<Material> h_materials;
    std::vector<Point3> h_vertices;
    std::vector<Direction3> h_normals;
};

struct DeviceScene {
    // CAMERA & SETTINGS
    Point3     camera_pos;
    Direction3 camera_fwd;
    Direction3 camera_up;
    Direction3 camera_right;
    float      camera_fov_ha;

    Color background;
    Color ambient_light;
    int max_depth;

    // LIGHTS
    DeviceLight* d_lights;
    int    num_lights;

    // PRIMITIVES
    Sphere* d_spheres;
    int     num_spheres;
    
    // [BVH] Sphere BVH Array
    BVHNode* d_sphereBVH; 

    Triangle* d_triangles;
    int       num_triangles;

    // [BVH] Triangle BVH Array
    BVHNode* d_triangleBVH; 

    // MATERIALS
    Material* d_materials;
    int       num_materials;

    // TRIANGLE VERTEX DATA
    Point3* d_vertices;
    Direction3* d_normals;
};

DeviceScene* parseSceneFile(const std::string &filename,
                     int &img_width,
                     int &img_height,
                     std::string &imgName);
