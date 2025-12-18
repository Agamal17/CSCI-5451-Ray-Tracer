# CSCI-5451 Ray Tracer (Base)

This repository contains the base sequential ray tracer implementation.

## Steps to compile and test on Plate server

1. Login to Plate
   ```bash
   ssh <your-username>@csel-plate01.cselabs.umn.edu
   ```

2. Clone the repository
   ```bash
   git clone https://github.com/Agamal17/CSCI-5451-Ray-Tracer.git
   ```

3. Navigate to the project directory
   ```bash
   cd CSCI-5451-Ray-Tracer
   ```

4. Compile the code
   ```bash
   g++ -std=c++17 -O2 main.cpp rayTrace.cpp scene.cpp lighting.cpp intersect.cpp primitive.cpp -I Include -o raytracer
   ```

5. Run a quick test (recommended, 6 minutes)
   ```bash
   ./raytracer Tests/InterestingScences/arm-top.txt
   ```

6. Run a complex scene (slow, takes hours of time)
   ```bash
   ./raytracer Tests/InterestingScences/dragon.txt
   ```
