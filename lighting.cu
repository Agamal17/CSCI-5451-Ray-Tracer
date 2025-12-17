#include <cmath> // For std::pow, std::max, M_PI
#include "Include/rayTrace.cuh"
#include "Include/lighting.cuh"
#include "Include/intersect.cuh" // Assuming FindIntersection is now __device__

static constexpr double EPS = 1e-4;

// --- Combined Device Contribution Function (The New Core Logic) ---

__device__ Color calculateContribution(
    const Direction3 N,
    const Direction3 L,
    const Direction3 V,
    const Color attenuated_color,
    const Material material)
{
    Color final_color(0, 0, 0);

    // Diffuse (Lambertian)
    // Use fmax/fmin or c++ std::max/std::min depending on your NVCC setup
    double NdotL = max(0.0, dot(N, L));
    final_color += material.diffuse * attenuated_color * NdotL;

    // Specular (Blinn–Phong)
    Direction3 H = (V + L).normalized();
    double NdotH = max(0.0, dot(N, H));
    final_color += material.specular * attenuated_color *
                   pow(NdotH, material.ns);

    return final_color;
}

__device__ Color getLightContribution(
    const DeviceScene* d_scene,
    const DeviceLight light,
    const Ray ray,
    const HitInfo hit)
{
    // Retrieve the material from the device array using the index
    const Material& material = d_scene->d_materials[hit.material_index];

    // Common vectors
    Direction3 N = hit.normal.normalized();
    Direction3 V = (-ray.dir).normalized();     // surface → camera
    Point3 p = hit.point + N * EPS;

    // Use the switch statement as the device-compatible replacement for virtual dispatch
    switch (light.type) {

        case DIRECTIONAL_LIGHT: {
            // *** FIX: Use light.direction_data instead of light.directional.direction ***
            Direction3 L = (-light.direction_data).normalized(); // surface → light

            Ray shadowRay(p, L);
            HitInfo shadowHit;

            if (!FindIntersection(d_scene, shadowRay, &shadowHit)) {
                return calculateContribution(N, L, V, light.color, material);
            }
            break; // Shadowed
        }

        case POINT_LIGHT: {
            // *** FIX: Use light.position_data instead of light.point.position ***
            Direction3 toLight = light.position_data - p;
            double light_distance = toLight.length();
            Direction3 L = toLight.normalized();    // surface → light

            Ray shadowRay(p, L);
            HitInfo shadowHit;

            // Check for shadow and distance
            if (!(FindIntersection(d_scene, shadowRay, &shadowHit) &&
                  shadowHit.distance < light_distance)) {

                Color attenuated_color = light.color / pow(light_distance, 2);
                return calculateContribution(N, L, V, attenuated_color, material);
            }
            break; // Shadowed
        }

        case SPOT_LIGHT: {
            // *** FIX: Use light.position_data instead of light.spot.position ***
            Direction3 toLight = light.position_data - p;
            double light_distance = toLight.length();
            Direction3 L = toLight.normalized();

            Ray shadowRay(p, L);
            HitInfo shadowHit;

            if (FindIntersection(d_scene, shadowRay, &shadowHit) &&
                shadowHit.distance < light_distance) {
                break; // Shadowed
            }

            // Angle check
            // *** FIX: Use light.spot_direction, light.angle1, light.angle2 ***
            Direction3 spotDir = light.spot_direction.normalized();

            double cosHitAngle = dot((-toLight).normalized(), spotDir);
            // Use fmax/fmin or std::max/std::min depending on your NVCC setup
            double hitAngle = acos(cosHitAngle) * 180.0 / M_PI;

            if (hitAngle > light.angle2) {
                break; // Outside outer cone
            }

            double falloff = 1.0;
            if (hitAngle > light.angle1) {
                double t = (hitAngle - light.angle1) / (light.angle2 - light.angle1);
                falloff = max(0.0, 1.0 - t);
            }

            Color attenuated_color = (light.color / pow(light_distance, 2)) * falloff;
            return calculateContribution(N, L, V, attenuated_color, material);
        }
    }

    return Color(0, 0, 0); // Default return for shadowed or outside cone
}