#include "Include/intersect.cuh"
#include "Include/lighting.cuh"
#include "Include/rayTrace.cuh"
#include <cmath>

__device__ Color rayTrace(const DeviceScene* d_scene, Ray initial_ray, const int max_depth) {
    Color accumulated_color(0, 0, 0);
    Color throughput(1, 1, 1); // Starts at full strength
    Ray current_ray = initial_ray;

    for (int depth = 0; depth < max_depth; ++depth) {
        HitInfo hit;

        // 1. Find the nearest object
        if (!FindIntersection(d_scene, current_ray, &hit)) {
            // If we miss, add background color (weighted by throughput) and stop
            accumulated_color += throughput * d_scene->background;
            break;
        }

        const Material& material = d_scene->d_materials[hit.material_index];

        // 2. Direct Lighting (Local Contribution)
        // We calculate ambient + direct light from all light sources here
        Color local_color = material.ambient * d_scene->ambient_light;
        for (int i = 0; i < d_scene->num_lights; ++i) {
            local_color += getLightContribution(d_scene, d_scene->d_lights[i], current_ray, hit);
        }

        // Add local contribution to our final image
        accumulated_color += throughput * local_color;

        // 3. Prepare for the next bounce
        // In a simple Whitted-style tracer, we usually follow either reflection or refraction.
        // If your material is both, you'd technically need a stack, but most GPU tracers
        // choose one (stochastic) or prioritize the dominant one to keep it truly iterative.

        if (material.trans.magnitude() > 0.0001) { // Material is refractive/transparent
            throughput = throughput * material.trans;
            current_ray = Refract(d_scene, current_ray, hit);
        }
        else if (material.specular.magnitude() > 0.0001) { // Material is reflective
            throughput = throughput * material.specular;
            current_ray = Reflect(current_ray, hit);
        }
        else {
            // Diffuse material with no secondary rays—terminate the path
            break;
        }

        // Russian Roulette or Epsilon check: if throughput is nearly zero, stop early
        if (throughput.magnitude() < 0.0001) break;
    }

    return accumulated_color;
}


__device__ Ray Reflect(Ray ray, HitInfo hit){
    Direction3 d = ray.dir.normalized();
    Direction3 n = hit.normal.normalized();

    Direction3 reflected_dir = d - 2.0 * dot(d, n) * n;

    return Ray(hit.point + reflected_dir * 0.001, reflected_dir.normalized());
}

__device__ Ray Refract(const DeviceScene* scene, Ray ray, HitInfo hit){
    Direction3 I = ray.dir.normalized();
    Direction3 N = hit.normal.normalized();

    float ior = scene->d_materials[hit.material_index].ior;
    float cos_theta = dot(I, N);
    float eta;
    Direction3 n_eff;

    if (cos_theta > 0) {
        eta = ior;
        n_eff = -1.0 * N;
    } else {
        eta = 1.0 / ior;
        n_eff = N;
        cos_theta = -cos_theta;
    }

    float k = 1.0 - eta * eta * (1.0 - cos_theta * cos_theta);

    if (k < 0) {
        return Reflect(ray, hit); 
    }

    Direction3 T = eta * I + (eta * cos_theta - std::sqrt(k)) * n_eff;

    return Ray(hit.point + T * 0.001, T.normalized());
}
