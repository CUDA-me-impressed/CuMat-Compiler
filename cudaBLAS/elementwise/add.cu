/*
    This file consists of CUDA code which is compiled with the CuMat
    program and linked in with the output with clang.

    Most of this is just setup for llvm with cuBLAS so that we can
    call the functions and have them return CuMat data (row major)
*/
#include <cstring>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "../utils/headers.hpp"

extern "C" void CuMatAddMatrixD(HeaderD* matHeaderA, HeaderD* matHeaderB, HeaderD* matHeaderRes, long len){
    //Temporary extraction of values
    double* matA;
    double* matB;
    double* matRes;
    matA = matHeaderA->data;
    matB = matHeaderB->data;
    matRes = matHeaderRes->data;

    // Pointers for the various kernel vars
    double *a, *b, *res;

    // Length calculations
    size_t matSize = len*sizeof(double);

    // Allocate on device
    cudaMallocManaged(&a, matSize);
    cudaMallocManaged(&b, matSize);
    cudaMallocManaged(&res, matSize);

    // Copy from host to device
    cublasSetVector(len, sizeof(double), matA, 1, a, 1);
    cublasSetVector(len, sizeof(double), matB, 1, b, 1);

    // Create cublas handler
    cublasHandle_t h;
    cublasCreate(&h);

    // Carry out addition
    const double scale = 1;
    cublasDaxpy(h, len, &scale, a,1, b, 1);
    // Copy vector off gpu
    cublasGetVector(len, sizeof(double), b, 1, res, 1);

    // Copy the results out of device memory
    cudaMemcpy(matRes, res, matSize, cudaMemcpyDeviceToHost);

    // Destroy cublas
    cublasDestroy(h);

    // Free device memory
    cudaFree(a);
    cudaFree(b);
}

// Device function
__global__ void CuMatAddMatrixIKernel(long* A, long* B, long * res, long len){
    long index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < len){
        res[index] = A[index] + B[index];
    }
}


extern "C" void CuMatAddMatrixI(HeaderI* matHeaderA, HeaderI* matHeaderB, HeaderI* matHeaderRes, long len){
     //Temporary extraction of values
    long* matA;
    long* matB;
    long* matRes;
    matA = matHeaderA->data;
    matB = matHeaderB->data;
    matRes = matHeaderRes->data;

    long* d_A; long *d_B; long * d_Res;
    size_t size = len*sizeof(long);
    // Allocate memory for CUDA
    cudaMallocManaged(&d_A, size);
    cudaMallocManaged(&d_B, size);
    cudaMallocManaged(&d_Res, size);

    // Copy over the matricies into device memory
    cudaMemcpy(d_A, matA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, matB, size, cudaMemcpyHostToDevice);

    // Set the number of threads per block and grid size
    int threadsPerBlock = 256;
    int blocksPerGrid = ((len) + threadsPerBlock -1) / threadsPerBlock;

    // Call the kernel
    CuMatAddMatrixIKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Res, len);

    // Synchronise before copying
    cudaDeviceSynchronize();

    // Copy the results out of device memory
    auto result = cudaMemcpy(matRes, d_Res, size, cudaMemcpyDeviceToHost);

    if(result != cudaSuccess){
        printf("GPUassert: %s\n", cudaGetErrorString(result));
    }

    // Free up cuda malloc
    cudaFree(&d_A);
    cudaFree(&d_B);
    cudaFree(&d_Res);
}
