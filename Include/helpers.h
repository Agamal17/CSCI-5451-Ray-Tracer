#pragma once

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR: %s - %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
};

// Helper function to free the memory allocated for the DeviceScene on the GPU
inline void freeDeviceScene(DeviceScene* d_scene_ptr) {
    if (!d_scene_ptr) return;

    // We must first copy the structure back to the host to access the internal pointers.
    DeviceScene h_d_scene;
    cudaMemcpy(&h_d_scene, d_scene_ptr, sizeof(DeviceScene), cudaMemcpyDeviceToHost);

    // Free all internal arrays allocated on the device
    if (h_d_scene.d_lights)    cudaFree(h_d_scene.d_lights);
    if (h_d_scene.d_spheres)   cudaFree(h_d_scene.d_spheres);
    if (h_d_scene.d_triangles) cudaFree(h_d_scene.d_triangles);
    if (h_d_scene.d_materials) cudaFree(h_d_scene.d_materials);
    if (h_d_scene.d_vertices)  cudaFree(h_d_scene.d_vertices);
    if (h_d_scene.d_normals)   cudaFree(h_d_scene.d_normals);

    // Finally, free the pointer to the structure itself
    cudaFree(d_scene_ptr);
}