#pragma once
#include <vector>
#include <string>
#include "types.h"
#include "primitive.h"
#include "lighting.h"

// ----------------- Device Scene -----------------
// This struct matches the Scene data but uses raw pointers/arrays
// instead of std::vector, allowing it to be passed to CUDA kernels.
struct SceneData {
    // Camera
    Point3     camera_pos;
    Direction3 camera_fwd;
    Direction3 camera_up;
    Direction3 camera_right;
    double     camera_fov_ha;

    // Global settings
    Color background;
    Color ambient_light;
    int max_depth;

    // Lights (Array of objects)
    // Note: Light struct must be flattened (non-polymorphic) for this to work on GPU
    Light* lights;
    int numLights;

    // Primitives (Arrays of objects)
    Sphere* spheres;
    int numSpheres;

    Triangle* triangles;
    int numTriangles;

    Material* materials;
    int numMaterials;

    // Triangle data (Raw arrays)
    Point3* vertices;
    int numVertices;
    Direction3* normals;
    int numNormals;
};

// ----------------- Host Scene -----------------
// Used for parsing and initial setup on CPU
struct Scene {
    // camera
    Point3     camera_pos;
    Direction3 camera_fwd;
    Direction3 camera_up;
    Direction3 camera_right;
    double      camera_fov_ha;

    // global settings
    Color background;
    Color ambient_light;
    int max_depth;

    std::vector<Light*> lights;

    // primitives
    std::vector<Sphere*>   spheres;
    std::vector<Triangle*>   triangles;

    std::vector<Material*>   materials;

    // triangle data
    std::vector<Point3>     vertices;  // positions
    std::vector<Direction3> normals;   // per-vertex normals
};

// parse scene file and fill scene + output image info
Scene parseSceneFile(const std::string &filename,
                     int &img_width,
                     int &img_height,
                     std::string &imgName);
