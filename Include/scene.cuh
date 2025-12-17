#pragma once
#include <vector>
#include <string>
#include "types.cuh"
#include "primitive.cuh"
#include "lighting.cuh"

// ----------------- Scene -----------------
struct HostSceneTemp {
    // This is the collection point for all dynamic data during parsing
    std::vector<DeviceLight> h_lights;
    std::vector<Sphere> h_spheres;
    std::vector<Triangle> h_triangles;
    std::vector<Material> h_materials;
    std::vector<Point3> h_vertices;
    std::vector<Direction3> h_normals;
};

struct DeviceScene {
    // CAMERA & SETTINGS (Simple data, can be passed by value)
    Point3     camera_pos;
        Direction3 camera_fwd;
    Direction3 camera_up;
    Direction3 camera_right;
    float      camera_fov_ha;

    Color background;
    Color ambient_light;
    int max_depth;

    // LIGHTS (Replaced std::vector<Light*> with flat C-array pointer)
    DeviceLight* d_lights;         // Pointer to the array of Light structures on the DEVICE
    int    num_lights;       // Number of elements in the d_lights array

    // PRIMITIVES
    Sphere* d_spheres;       // Pointer to the array of Sphere structures on the DEVICE
    int     num_spheres;

    Triangle* d_triangles;   // Pointer to the array of Triangle structures on the DEVICE
    int       num_triangles;

    // MATERIALS
    Material* d_materials;   // Pointer to the array of Material structures on the DEVICE
    int       num_materials;

    // TRIANGLE VERTEX DATA (Contiguous, indexed arrays)
    Point3* d_vertices;  // Pointer to the array of vertex positions on the DEVICE
    Direction3* d_normals;   // Pointer to the array of normals on the DEVICE
};

// parse scene file and fill scene + output image info
DeviceScene* parseSceneFile(const std::string &filename,
                     int &img_width,
                     int &img_height,
                     std::string &imgName);
