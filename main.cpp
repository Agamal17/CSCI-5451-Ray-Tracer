// For Visual Studio
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#endif

// MPI
#include <mpi.h>

// Image and ray tracer includes
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "Include/Image/image_lib.h"
#include "Include/ray.h"
#include "Include/rayTrace.h"
#include "Include/scene.h"

#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size = 0;
    int world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Check command-line arguments (only rank 0 prints the message)
    if (argc < 2) {
        if (world_rank == 0) {
            std::cout << "Usage: mpirun -np <procs> raytracer <scenefile>\n";
        }
        MPI_Finalize();
        return 0;
    }

    std::string sceneFileName = argv[1];
    int img_width, img_height;
    std::string imgName;

    // Each rank parses the same scene file (simple but sufficient)
    Scene scene = parseSceneFile(sceneFileName, img_width, img_height, imgName);

    // Each rank allocates an image; only rank 0 will finally write to disk
    Image outputImg(img_width, img_height);

    // Camera precomputations
    float imgW   = static_cast<float>(img_width);
    float imgH   = static_cast<float>(img_height);
    float halfW  = imgW / 2.0f;
    float halfH  = imgH / 2.0f;
    float d      = halfH / tanf(scene.camera_fov_ha * (M_PI / 180.0f));

    // Camera origin of the view plane: move forward by distance d
    Point3 cam_origin = scene.camera_pos - d * scene.camera_fwd;

    // Step in x-direction between neighboring pixels (column step)
    Direction3 step_x = -scene.camera_right;

    // Distribute rows across ranks
    int base_rows = img_height / world_size;
    int remainder = img_height % world_size;

    int local_rows = base_rows + (world_rank < remainder ? 1 : 0);
    int start_row  = world_rank * base_rows + std::min(world_rank, remainder);

    // Each pixel has 3 floats (RGB)
    int sendcount = local_rows * img_width * 3;
    std::vector<float> local_pixels(sendcount);

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Render rows assigned to this rank
    for (int lr = 0; lr < local_rows; ++lr) {
        int j = start_row + lr;  // global row index
        float v = halfH - static_cast<float>(j) + 0.5f;

        // Starting point on the view plane for column 0 of this row
        Point3 row_start = cam_origin
                         + v * scene.camera_up
                         + (halfW + 0.5f) * scene.camera_right;

        Point3 p = row_start;

        for (int i = 0; i < img_width; ++i) {
            Ray ray(scene.camera_pos, p - scene.camera_pos);
            Color result = rayTrace(ray, scene.max_depth, scene);

            int idx = (lr * img_width + i) * 3;
            local_pixels[idx + 0] = result.x;
            local_pixels[idx + 1] = result.y;
            local_pixels[idx + 2] = result.z;

            // Move to next pixel in the row
            p = p + step_x;
        }
    }

    double t1       = MPI_Wtime();
    double local_ms = (t1 - t0) * 1000.0;

    // Use the maximum across ranks as the global wall-clock time
    double global_ms = 0.0;
    MPI_Reduce(&local_ms, &global_ms, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    // Prepare recvcounts and displacements for Gatherv
    std::vector<int> recvcounts(world_size);
    std::vector<int> displs(world_size);

    for (int r = 0; r < world_size; ++r) {
        int rows_r     = base_rows + (r < remainder ? 1 : 0);
        recvcounts[r]  = rows_r * img_width * 3;

        int start_r    = r * base_rows + std::min(r, remainder);
        displs[r]      = start_r * img_width * 3;
    }

    std::vector<float> all_pixels;
    if (world_rank == 0) {
        all_pixels.resize(img_width * img_height * 3);
    }

    // Gather all pixel data to rank 0
    MPI_Gatherv(local_pixels.data(), sendcount, MPI_FLOAT,
                world_rank == 0 ? all_pixels.data() : nullptr,
                recvcounts.data(), displs.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // Rank 0 assembles final image and writes output file
    if (world_rank == 0) {
        for (int j = 0; j < img_height; ++j) {
            for (int i = 0; i < img_width; ++i) {
                int idx = (j * img_width + i) * 3;
                Color &pixel = outputImg.getPixel(i, j);
                pixel.x = all_pixels[idx + 0];
                pixel.y = all_pixels[idx + 1];
                pixel.z = all_pixels[idx + 2];
            }
        }

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\n[TIMING][MPI] total (max over ranks): "
                  << global_ms << " ms\n\n";

        outputImg.write(imgName.c_str());
    }

    // Free scene objects (each rank frees its own copy)
    for (Sphere* s : scene.spheres)     delete s;
    for (Triangle* t : scene.triangles) delete t;
    for (Material* m : scene.materials) delete m;
    for (Light* l : scene.lights)       delete l;

    MPI_Finalize();
    return 0;
}
