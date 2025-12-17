#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <vector>
#include "Include/scene.cuh"
#include "Include/helpers.h"

// Trim leading and trailing whitespace from a string
static inline void trimInPlace(std::string &s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
        s.clear();
        return;
    }
    size_t e = s.find_last_not_of(" \t\r\n");
    s = s.substr(b, e - b + 1);
}

DeviceScene* parseSceneFile(const std::string &filename,
                     int &img_width,
                     int &img_height,
                     std::string &imgName) {

    // 1. Temporary Host Structures
    HostSceneTemp h_temp;          // Dynamic collections (std::vectors)
    DeviceScene h_d_scene;         // Host copy of the final structure to be transferred

    // Default camera and global settings (written to h_d_scene)
    h_d_scene.camera_pos    = Direction3(0, 0, 0);
    h_d_scene.camera_fwd    = Direction3(0, 0, -1);
    h_d_scene.camera_up     = Direction3(0, 1, 0);
    h_d_scene.camera_fov_ha = 45.0;

    h_d_scene.background    = Color(0, 0, 0);
    h_d_scene.ambient_light = Color(0, 0, 0);
    h_d_scene.max_depth = 5;

    // Default image parameters
    img_width  = 640;
    img_height = 480;
    imgName    = "raytraced.bmp";

    // Default material: 0 0 0  1 1 1  0 0 0  5  0 0 0  1
    // Store materials directly in the vector.
    Material default_material;
    default_material.ambient  = Color(0, 0, 0);
    default_material.diffuse  = Color(1, 1, 1);
    default_material.specular = Color(0, 0, 0);
    default_material.ns       = 5.0;
    default_material.trans    = Color(0, 0, 0);
    default_material.ior      = 1.0;
    h_temp.h_materials.push_back(default_material);

    // Track the index of the currently active material
    size_t current_material_index = 0;

    int max_vertices = 0;
    int max_normals  = 0;

    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Cannot open scene file: " << filename << std::endl;
        return nullptr; // Return nullptr on failure
    }

    std::string line;
    while (std::getline(in, line)) {
        trimInPlace(line);
        if (line.empty()) continue;
        if (line[0] == '#') continue; // comment line

        std::string key, rest;
        size_t colonPos = line.find(':');
        if (colonPos != std::string::npos) {
            key  = line.substr(0, colonPos);
            rest = line.substr(colonPos + 1);
        } else {
            std::stringstream ss(line);
            ss >> key;
            std::getline(ss, rest);
        }
        trimInPlace(key);
        trimInPlace(rest);
        std::stringstream ss(rest);

        if (key == "film_resolution") {
            ss >> img_width >> img_height;
        } else if (key == "output_image") {
            ss >> imgName;
            trimInPlace(imgName);
            if (!imgName.empty() && imgName.front() == '"' && imgName.back() == '"') {
                imgName = imgName.substr(1, imgName.size() - 2);
            }
        } else if (key == "camera_pos") {
            ss >> h_d_scene.camera_pos.x >> h_d_scene.camera_pos.y >> h_d_scene.camera_pos.z;
        } else if (key == "camera_fwd") {
            ss >> h_d_scene.camera_fwd.x >> h_d_scene.camera_fwd.y >> h_d_scene.camera_fwd.z;
        } else if (key == "camera_up") {
            ss >> h_d_scene.camera_up.x >> h_d_scene.camera_up.y >> h_d_scene.camera_up.z;
        } else if (key == "camera_fov_ha") {
            ss >> h_d_scene.camera_fov_ha;
        } else if (key == "background") {
            ss >> h_d_scene.background.r >> h_d_scene.background.g >> h_d_scene.background.b;
        } else if (key == "ambient_light") {
            ss >> h_d_scene.ambient_light.r >> h_d_scene.ambient_light.g >> h_d_scene.ambient_light.b;
        } else if (key == "material") {
            Material new_material;
            ss >> new_material.ambient.r >> new_material.ambient.g >> new_material.ambient.b
               >> new_material.diffuse.r >> new_material.diffuse.g >> new_material.diffuse.b
               >> new_material.specular.r >> new_material.specular.g >> new_material.specular.b
               >> new_material.ns
               >> new_material.trans.r >> new_material.trans.g >> new_material.trans.b
               >> new_material.ior;

            h_temp.h_materials.push_back(new_material);
            current_material_index = h_temp.h_materials.size() - 1; // Update active material index

        } else if (key == "sphere") {
            double x, y, z, r;
            ss >> x >> y >> z >> r;
            Sphere s;
            s.center      = Point3(x, y, z);
            s.radius      = r;
            s.material_index = current_material_index; // Use index instead of pointer
            h_temp.h_spheres.push_back(s);

        } else if (key == "max_vertices") {
            ss >> max_vertices;
            if (max_vertices < 0) max_vertices = 0;
            h_temp.h_vertices.reserve(max_vertices);
        } else if (key == "max_normals") {
            ss >> max_normals;
            if (max_normals < 0) max_normals = 0;
            h_temp.h_normals.reserve(max_normals);
        } else if (key == "vertex") {
            double x, y, z;
            ss >> x >> y >> z;
            if (max_vertices > 0 && (int)h_temp.h_vertices.size() >= max_vertices) {
                std::cerr << "Warning: more vertices than max_vertices in " << filename << std::endl;
            }
            h_temp.h_vertices.push_back(Point3(x, y, z));
        } else if (key == "normal") {
            double x, y, z;
            ss >> x >> y >> z;
            if (max_normals > 0 && (int)h_temp.h_normals.size() >= max_normals) {
                std::cerr << "Warning: more normals than max_normals in " << filename << std::endl;
            }
            h_temp.h_normals.push_back(Direction3(x, y, z));
        } else if (key == "triangle") {
            int v0, v1, v2;
            ss >> v0 >> v1 >> v2;

            Triangle t;
            // Use indices to retrieve vertices from the collected vector
            if (v0 < h_temp.h_vertices.size() && v1 < h_temp.h_vertices.size() && v2 < h_temp.h_vertices.size()) {
                t.v1 = h_temp.h_vertices[v0];
                t.v2 = h_temp.h_vertices[v1];
                t.v3 = h_temp.h_vertices[v2];

                // Calculate face normal
                Direction3 fn = cross(t.v2 - t.v1, t.v3 - t.v1).normalized();
                t.triPlane = Direction3(fn);

                t.n1 = t.n2 = t.n3 = t.triPlane; // Flat shading normals

                t.flat = true;

                t.material_index = current_material_index;
                h_temp.h_triangles.push_back(t);
            } else {
                std::cerr << "Error: Invalid vertex index in triangle." << std::endl;
            }

        } else if (key == "normal_triangle") {
            int v0, v1, v2, n0, n1, n2;
            ss >> v0 >> v1 >> v2 >> n0 >> n1 >> n2;

            Triangle t;
             // Check indices before assignment
            if (v0 < h_temp.h_vertices.size() && v1 < h_temp.h_vertices.size() && v2 < h_temp.h_vertices.size() &&
                n0 < h_temp.h_normals.size() && n1 < h_temp.h_normals.size() && n2 < h_temp.h_normals.size()) {

                t.v1 = h_temp.h_vertices[v0];
                t.v2 = h_temp.h_vertices[v1];
                t.v3 = h_temp.h_vertices[v2];

                t.n1 = h_temp.h_normals[n0];
                t.n2 = h_temp.h_normals[n1];
                t.n3 = h_temp.h_normals[n2];

                Direction3 fn = cross(t.v2 - t.v1, t.v3 - t.v1).normalized();
                t.triPlane = Direction3(fn);

                t.flat = false;

                t.material_index = current_material_index;
                h_temp.h_triangles.push_back(t);
            } else {
                std::cerr << "Error: Invalid vertex or normal index in normal_triangle." << std::endl;
            }

        } else if (key == "directional_light") {
            double r, g, b, x, y, z;
            ss >> r >> g >> b >> x >> y >> z;
            // Host side code during parsing:
            DeviceLight new_light;
            new_light.type = DIRECTIONAL_LIGHT;
            new_light.color = Color(r, g, b);
            new_light.direction_data = Direction3(x, y, z);

            h_temp.h_lights.push_back(new_light);
        } else if (key == "point_light") {
            double r, g, b, x, y, z;
            ss >> r >> g >> b >> x >> y >> z;
            DeviceLight new_light;
            new_light.type = POINT_LIGHT;
            new_light.color = Color(r, g, b);
            new_light.position_data = Point3(x, y, z);

            h_temp.h_lights.push_back(new_light);
        } else if (key == "spot_light") {
            double r, g, b, x, y, z, dir_x, dir_y, dir_z, angle1, angle2;
            ss >> r >> g >> b >> x >> y >> z >> dir_x >> dir_y >> dir_z >> angle1 >> angle2;

            DeviceLight new_light;
            new_light.type = SPOT_LIGHT;
            new_light.color = Color(r, g, b);
            new_light.position_data = Point3(x, y, z);
            new_light.spot_direction = Direction3(dir_x, dir_y, dir_z);
            new_light.angle1 = angle1;
            new_light.angle2 = angle2;

            h_temp.h_lights.push_back(new_light);
        } else if (key == "max_depth"){
            ss >> h_d_scene.max_depth;
        } else {
            std::cerr << "Warning: unknown key " << key << " in " << filename << std::endl;
        }
    }

    // Final calculations for camera orientation
    h_d_scene.camera_right = cross(h_d_scene.camera_up, h_d_scene.camera_fwd).normalized();
    h_d_scene.camera_up = cross(h_d_scene.camera_fwd, h_d_scene.camera_right).normalized();
    h_d_scene.camera_fwd = h_d_scene.camera_fwd.normalized();


    // --- 3. CUDA ALLOCATION AND TRANSFER (Deep Copy) ---

    // 3.1. Allocate and Copy Lights
    h_d_scene.num_lights = h_temp.h_lights.size();
    if (h_d_scene.num_lights > 0) {
        checkCuda(cudaMalloc((void**)&h_d_scene.d_lights, h_d_scene.num_lights * sizeof(DeviceLight)), "Lights Malloc");
        checkCuda(cudaMemcpy(h_d_scene.d_lights, h_temp.h_lights.data(),
                   h_d_scene.num_lights * sizeof(DeviceLight), cudaMemcpyHostToDevice), "Lights Memcpy");
    } else {
        h_d_scene.d_lights = nullptr;
    }

    // 3.2. Allocate and Copy Spheres
    h_d_scene.num_spheres = h_temp.h_spheres.size();
    if (h_d_scene.num_spheres > 0) {
        checkCuda(cudaMalloc((void**)&h_d_scene.d_spheres, h_d_scene.num_spheres * sizeof(Sphere)), "Spheres Malloc");
        checkCuda(cudaMemcpy(h_d_scene.d_spheres, h_temp.h_spheres.data(),
                   h_d_scene.num_spheres * sizeof(Sphere), cudaMemcpyHostToDevice), "Spheres Memcpy");
    } else {
        h_d_scene.d_spheres = nullptr;
    }

    // 3.3. Allocate and Copy Triangles
    h_d_scene.num_triangles = h_temp.h_triangles.size();
    if (h_d_scene.num_triangles > 0) {
        checkCuda(cudaMalloc((void**)&h_d_scene.d_triangles, h_d_scene.num_triangles * sizeof(Triangle)), "Triangles Malloc");
        checkCuda(cudaMemcpy(h_d_scene.d_triangles, h_temp.h_triangles.data(),
                   h_d_scene.num_triangles * sizeof(Triangle), cudaMemcpyHostToDevice), "Triangles Memcpy");
    } else {
        h_d_scene.d_triangles = nullptr;
    }

    // 3.4. Allocate and Copy Materials
    h_d_scene.num_materials = h_temp.h_materials.size();
    if (h_d_scene.num_materials > 0) {
        checkCuda(cudaMalloc((void**)&h_d_scene.d_materials, h_d_scene.num_materials * sizeof(Material)), "Materials Malloc");
        checkCuda(cudaMemcpy(h_d_scene.d_materials, h_temp.h_materials.data(),
                   h_d_scene.num_materials * sizeof(Material), cudaMemcpyHostToDevice), "Materials Memcpy");
    } else {
        h_d_scene.d_materials = nullptr;
    }

    // 3.5. Allocate and Copy Vertices
    if (h_temp.h_vertices.size() > 0) {
        checkCuda(cudaMalloc((void**)&h_d_scene.d_vertices, h_temp.h_vertices.size() * sizeof(Point3)), "Vertices Malloc");
        checkCuda(cudaMemcpy(h_d_scene.d_vertices, h_temp.h_vertices.data(),
                   h_temp.h_vertices.size() * sizeof(Point3), cudaMemcpyHostToDevice), "Vertices Memcpy");
    } else {
        h_d_scene.d_vertices = nullptr;
    }

    // 3.6. Allocate and Copy Normals
    if (h_temp.h_normals.size() > 0) {
        checkCuda(cudaMalloc((void**)&h_d_scene.d_normals, h_temp.h_normals.size() * sizeof(Direction3)), "Normals Malloc");
        checkCuda(cudaMemcpy(h_d_scene.d_normals, h_temp.h_normals.data(),
                   h_temp.h_normals.size() * sizeof(Direction3), cudaMemcpyHostToDevice), "Normals Memcpy");
    } else {
        h_d_scene.d_normals = nullptr;
    }

    // --- 4. Final Shallow Copy of the Scene Struct ---
    // Copy the h_d_scene struct (which holds simple values and all the D_evice pointers)
    // to a persistent location on the GPU.
    DeviceScene* d_scene_ptr;
    checkCuda(cudaMalloc((void**)&d_scene_ptr, sizeof(DeviceScene)), "Final Scene Malloc");
    checkCuda(cudaMemcpy(d_scene_ptr, &h_d_scene, sizeof(DeviceScene), cudaMemcpyHostToDevice), "Final Scene Memcpy");

    return d_scene_ptr;
}