# CSCI-5451 Ray Tracer (CUDA)

This repository contains the CUDA-Parallel ray tracer implementation.

## Steps to compile and test on CUDA server

1. Login to CUDA
   ```bash
   ssh <your-username>@csel-cuda-04.cselabs.umn.edu
   ```

2. Clone the CUDA branch only
   ```bash
   git clone --branch CUDA --single-branch https://github.com/Agamal17/CSCI-5451-Ray-Tracer.git
   ```

3. Navigate to the project directory
   ```bash
   cd CSCI-5451-Ray-Tracer
   ```
4. Cross Verify branch using command
   ```bash
   git branch
   ```

5. Compile the CUDA version
   ```bash
   nvcc -D_USE_MATH_DEFINES -Xptxas -O3 -arch=sm_75 scene.cpp main.cu primitive.cu lighting.cu intersect.cu rayTrace.cu rayTraceKernel.cu -o raytracer_cuda -rdc=true
   ```

6. Run a quick test (recommended)
   ```bash
   ./raytracer_cuda Tests/InterestingScenes/arm-top.txt
   ```

7. Run a complex scene
   ```bash
   ./raytracer_cuda Tests/InterestingScenes/dragon.txt
   ```
