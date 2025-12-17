#pragma once
#include "types.cuh"
#include "ray.cuh"
#include "scene.cuh"

struct DeviceScene;

// --- ENUM: Replaces Virtual Functions ---
enum LightType {
    DIRECTIONAL_LIGHT,
    POINT_LIGHT,
    SPOT_LIGHT
};

// --- DEVICE LIGHT STRUCTURE (Replaces Light Base Class) ---
// This single structure is what is copied to the GPU as an array.
struct DeviceLight {
    LightType type;
    Color color;

    // Directional Light Data
    Direction3 direction_data; // Only valid if type == DIRECTIONAL_LIGHT

    // Point Light Data
    Point3 position_data;      // Valid for POINT_LIGHT or SPOT_LIGHT

    // Spot Light Data
    Direction3 spot_direction; // Only valid if type == SPOT_LIGHT
    float angle1;             // Only valid if type == SPOT_LIGHT
    float angle2;             // Only valid if type == SPOT_LIGHT
};

// --- DEVICE FUNCTIONS (Called from Kernel) ---
// Note: These need to be declared here and implemented in the .cu file
__device__ Color getLightContribution(
    const DeviceScene* d_scene,
    const DeviceLight light,
    const Ray ray,
    const HitInfo hit);