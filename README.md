CSCI-5451 Ray Tracer (MPI)
This repository contains the MPI-parallel ray tracer implementation.

Steps to compile and test on Plate server
Login to Plate

ssh <your-username>@csel-plate01.cselabs.umn.edu
(plate02–plate04 are also fine.)

Clone the repository

git clone https://github.com/<team-repo>/CSCI-5451-Ray-Tracer.git
cd CSCI-5451-Ray-Tracer

Compile the MPI version

mpicxx -O3 -std=c++17 main.cpp rayTrace.cpp scene.cpp lighting.cpp intersect.cpp primitive.cpp \
    -IInclude -IInclude/Image -o ray_mpi

Run a quick scene (np = 64)

mpirun -np 64 ./ray_mpi Tests/TriangleExamples/triangle.txt

Run a large test (np = 64)

mpirun -np 64 ./ray_mpi Tests/TriangleExamples/test_reasonable.txt



(Optional) Batch timing with np = 64

./run_mpi_batch.sh


