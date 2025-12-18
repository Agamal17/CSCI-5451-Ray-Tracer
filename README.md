# CSCI-5451 Ray Tracer (OpenMP)

This repository contains the OpenMP-Parallel ray tracer implementation.

## Steps to compile and test on Plate server

1. Login to Plate
   ```bash
   ssh <your-username>@csel-plate01.cselabs.umn.edu
   ```

2. Clone the repository
   ```bash
   git clone --branch OpenMP --single-branch https://github.com/Agamal17/CSCI-5451-Ray-Tracer.git
   ```

3. Navigate to the project directory
   ```bash
   cd CSCI-5451-Ray-Tracer
   ```

4. Compile the code
   ```bash
   g++ -std=c++17 -O2 -fopenmp main.cpp rayTrace.cpp scene.cpp lighting.cpp intersect.cpp primitive.cpp -I Include -o raytracer_omp
   ```

5. Run a quick test (recommended)
   ```bash
   ./raytracer_omp Tests/InterestingScenes/arm-top.txt
   ```

6. Run a complex scene
   ```bash
   ./raytracer_omp Tests/InterestingScenes/dragon.txt
   ```
