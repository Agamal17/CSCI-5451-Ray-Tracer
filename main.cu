#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES

// Images Lib includes:
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "Include/Image/image_lib.cuh"
#include "Include/ray.cuh"
#include "Include/rayTrace.cuh"
#include "Include/scene.cuh"
#include <iostream>
#include <string>
#include "Include/rayTraceKernel.cuh"
#include "Include/helpers.h"
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    // --- 0. Initialization ---
    if (argc < 2) {
        std::cout << "Usage: ./a.out scenefile\n";
        return 0;
    }

    // --- 1. Scene Parsing and Device Setup ---
    std::string sceneFileName = argv[1];
    int img_width, img_height;
    std::string imgName;

    // Parse file and allocate ALL necessary scene data onto the GPU
    DeviceScene* d_scene = parseSceneFile(sceneFileName, img_width, img_height, imgName);

    if (!d_scene) {
        std::cerr << "Failed to parse scene file or allocate device memory." << std::endl;
        return 1;
    }

    // Host Image for final output (Assuming output_img.pixels is the pointer to host data)
    Image output_img(img_width, img_height);

    // Device Image buffer (Allocated on GPU)
    Color* d_output_img;
    checkCuda(cudaMalloc((void**)&d_output_img, img_width * img_height * sizeof(Color)), "Output Image Malloc");

    // --- 3. Kernel Configuration and Launch ---
    dim3 threads_per_block(16, 16);
    dim3 num_blocks(
        (img_width + threads_per_block.x - 1) / threads_per_block.x,
        (img_height + threads_per_block.y - 1) / threads_per_block.y
    );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Launch the kernel
    rayTraceKernel<<<num_blocks, threads_per_block>>>(d_output_img, img_width, img_height, d_scene);

    // Record the stop event
    cudaEventRecord(stop);

    // Wait for the stop event to actually finish
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "[Timing][CUDA] total: " << milliseconds << " ms\n\n" << std::endl;

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Synchronize and check for errors that occurred inside the kernel
    checkCuda(cudaDeviceSynchronize(), "Kernel Execution and Synchronization");

    // --- 4. Final Data Transfer and Cleanup ---
    // Copy result from Device to Host
    checkCuda(cudaMemcpy(output_img.pixels, d_output_img,
                         img_width * img_height * sizeof(Color), cudaMemcpyDeviceToHost), "Result Memcpy");

    // Free the Device output buffer
    checkCuda(cudaFree(d_output_img), "Output Image Free");

    // !!! CRITICAL: Free all scene data allocated on the GPU !!!
    freeDeviceScene(d_scene);

    // --- 5. Output ---
    output_img.write(imgName.c_str());

    return 0;
}