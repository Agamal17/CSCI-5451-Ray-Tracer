# CSCI-5451 Ray Tracer (CUDA)

This repository contains the GPU-accelerated ray tracer implementation using CUDA.

## Steps to compile and test on CUDA server

1. Login to CUDA
   ```bash
   ssh <your-username>@csel-cuda-04.cselabs.umn.edu
   
2. Clone the repository
   ```bash
   git clone [https://github.com/Agamal17/CSCI-5451-Ray-Tracer.git](https://github.com/Agamal17/CSCI-5451-Ray-Tracer.git)
   ```
3. Navigate to the project directory
   ```bash
   cd CSCI-5451-Ray-Tracer
   ```
4. Compile the CUDA code
      ```bash
   nvcc -D_USE_MATH_DEFINES -Xptxas -O3 -arch=sm_75 scene.cpp main.cu primitive.cu lighting.cu intersect.cu rayTrace.cu rayTraceKernel.cu -o build/main -rdc=true
   ```

5. Run a quick test
   ```bash
    ./build/main Tests/InterestingScenes/arm-top.txt
   ```

6. Run a complex scene
   ```bash
    ./build/main Tests/InterestingScenes/dragon.txt
   ```