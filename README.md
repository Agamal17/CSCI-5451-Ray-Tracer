# CSCI-5451-Ray-Tracer

# CUDA implemntation of the ray tracer

Compilation command:


nvcc -D_USE_MATH_DEFINES -Xptxas -O3 -arch=sm_75 scene.cpp main.cu primitive.cu lighting.cu intersect.cu rayTrace.cu rayTraceKernel.cu -o build/main -rdc=true
