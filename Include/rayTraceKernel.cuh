#pragma once
#include "scene.cuh"
#include "Image/image_lib.cuh"

__global__ void rayTraceKernel(Color* outputImg, const int imgW, const int imgH, const DeviceScene* Scene);