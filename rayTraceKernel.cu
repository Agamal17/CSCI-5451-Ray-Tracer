#define _USE_MATH_DEFINES
#include "Include/types.cuh"
#include "Include/ray.cuh"
#include "Include/rayTraceKernel.cuh"
#include "Include/rayTrace.cuh"


__global__ void rayTraceKernel(Color* output_img, const int imgW, const int imgH, const DeviceScene* scene) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= imgW || y >= imgH) return;

    float u = imgW / 2 - x + 0.5;
    float v = imgH / 2 - y + 0.5;
    float d = (imgH / 2) / tan(scene->camera_fov_ha * M_PI / 180);
    Point3 p = scene->camera_pos - d * scene->camera_fwd + u * scene->camera_right + v * scene->camera_up;

    Ray ray(scene->camera_pos, p - scene->camera_pos);

    int pixelIndex = y * imgW + x;

    Color color = rayTrace(scene, ray, scene->max_depth);

    output_img[pixelIndex] = color;
}
