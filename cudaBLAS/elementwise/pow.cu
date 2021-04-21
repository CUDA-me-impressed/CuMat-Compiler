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
#include <math.h>
#include "../utils/headers.hpp"

// Device function
__global__ void CuMatPowMatrixDKernel(double* A, double* B, double * res, long len){
    long index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < len){
        res[index] = pow(A[index],B[index]);
    }
}

extern "C" void CuMatPowMatrixD(HeaderD* matHeaderA, HeaderD* matHeaderB, HeaderD* matHeaderRes, long len){
    double* matA;
    double* matB;
    double* matRes;
    matA = matHeaderA->data;
    matB = matHeaderB->data;
    matRes = matHeaderRes->data;
    double* d_A; double *d_B; double * d_Res;
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
    int blocksPerGrid = (len + threadsPerBlock -1) / threadsPerBlock;

    // Call the kernel
    CuMatPowMatrixDKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Res, len);

    // Synchronise before copying
    cudaDeviceSynchronize();

    // Copy the results out of device memory
    cudaMemcpy(matRes, d_Res, size, cudaMemcpyDeviceToHost);

    // Free up cuda malloc
    cudaFree(&d_A);
    cudaFree(&d_B);
    cudaFree(&d_Res);
}

// Device function
__global__ void CuMatPowMatrixIKernel(long* A, long* B, long * res, long len){
    long index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < len){
        res[index] = pow(A[index], B[index]);
    }
}


extern "C" void CuMatPowMatrixI(HeaderI* matHeaderA, HeaderI* matHeaderB, HeaderI* matHeaderRes, long len){
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
    int blocksPerGrid = (len + threadsPerBlock -1) / threadsPerBlock;

    // Call the kernel
    CuMatPowMatrixIKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Res, len);

    // Synchronise before copying
    cudaDeviceSynchronize();

    // Copy the results out of device memory
    cudaMemcpy(matRes, d_Res, size, cudaMemcpyDeviceToHost);

    // Free up cuda malloc
    cudaFree(&d_A);
    cudaFree(&d_B);
    cudaFree(&d_Res);
}