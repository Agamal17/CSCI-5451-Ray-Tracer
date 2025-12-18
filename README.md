# CSCI-5451 Ray Tracer (CUDA with BVH)

This repository contains the CUDA ray tracer implementation with Acceleration Data Structure BVH.

## Steps to compile and test on CUDA server

1. Login to CUDA
   ```bash
   ssh <your-username>@csel-cuda-04.cselabs.umn.edu
   ```

2. Clone the CUDA branch only
   ```bash
   git clone --branch CUDA-BVH --single-branch https://github.com/Agamal17/CSCI-5451-Ray-Tracer.git
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
   nvcc -D_USE_MATH_DEFINES -Xptxas -O3 -arch=sm_75 scene.cpp main.cu primitive.cu lighting.cu intersect.cu rayTrace.cu rayTraceKernel.cu -o raytracer_cudabvh -rdc=true
   ```

6. Run a quick test (recommended)
   ```bash
   ./raytracer_cudabvh Tests/InterestingScenes/arm-top.txt
   ```

7. Run a complex scene
   ```bash
   ./raytracer_cudabvh Tests/InterestingScenes/dragon.txt
   ```



<img width="640" height="480" alt="outdoor" src="https://github.com/user-attachments/assets/808d9dab-7f60-454f-ac4e-acd73b190478" />
<img width="1024" height="768" alt="foo" src="https://github.com/user-attachments/assets/8882e0e4-0e84-468b-bf19-404b8ab789bd" />
<img width="640" height="480" alt="spheres2" src="https://github.com/user-attachments/assets/e2db888b-e887-4735-9917-9f1447872891" />
<img width="1024" height="768" alt="bear" src="https://github.com/user-attachments/assets/09f5a7ad-d2aa-49b0-a715-79ca95c33017" />
<img width="640" height="480" alt="spot_sphere" src="https://github.com/user-attachments/assets/5891d400-c8f1-4b60-9fbb-301ad2f957c9" />
<img width="800" height="600" alt="watch_easy_mode" src="https://github.com/user-attachments/assets/ba932e9a-680c-4af0-937d-cc5bd89c81e7" />
<img width="1920" height="1080" alt="watch_blue_and_gold" src="https://github.com/user-attachments/assets/849b8be9-4707-4cb4-87cf-3c6ca61f8bf0" />
<img width="1920" height="1080" alt="sword" src="https://github.com/user-attachments/assets/cb3d92c9-0818-4189-9c1a-a8c806d39e59" />
<img width="1280" height="720" alt="spaceship" src="https://github.com/user-attachments/assets/2378e16f-8a38-42ca-9e66-f9afac71e787" />
<img width="800" height="600" alt="ShadowTest" src="https://github.com/user-attachments/assets/4ae5f20a-2aff-4d32-9c77-5ca32ce202f3" />
<img width="800" height="600" alt="reachingHand" src="https://github.com/user-attachments/assets/9ebd58dd-8c38-44a1-a0ca-360b0b3c9fd5" />
<img width="660" height="512" alt="plant" src="https://github.com/user-attachments/assets/d3456629-dfac-4a12-b7dc-9137814e5fff" />
<img width="800" height="600" alt="noLabel" src="https://github.com/user-attachments/assets/9b2e29d0-879b-43b4-b522-5678de91295b" />
<img width="640" height="480" alt="bottle" src="https://github.com/user-attachments/assets/48a2f063-bd97-440c-aa82-6188bc074862" />
<img width="1920" height="1080" alt="lily" src="https://github.com/user-attachments/assets/d50dd61a-a39d-4039-b451-94e7c93a8085" />
<img width="1280" height="720" alt="island" src="https://github.com/user-attachments/assets/d517e03d-dd33-4c8a-b29e-87e5596fa6d9" />
<img width="800" height="600" alt="gear" src="https://github.com/user-attachments/assets/7960e44d-9c74-453a-8981-74d5ffbc749f" />
<img width="512" height="384" alt="dragon" src="https://github.com/user-attachments/assets/014de38c-78b7-4b6b-ad14-a04b2311bcbf" />
<img width="1920" height="1080" alt="character" src="https://github.com/user-attachments/assets/2a52a9a8-f136-426e-9f2f-ab65722f82fa" />
<img width="1920" height="1080" alt="cat" src="https://github.com/user-attachments/assets/76d297a8-21fe-45d2-86f6-2cc1a064a2d6" />
<img width="400" height="600" alt="arm" src="https://github.com/user-attachments/assets/becc4db3-2209-47cf-86cb-59176070b7a2" />

