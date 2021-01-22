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

void CuMatAddMatrixD(double * matA, double * matB, double * matRes, long i, long j){
    // Pointers for the various kernel vars
    double *a, *b, *res;

    // Length calculations
    long N = i*j;
    size_t matSize = N*sizeof(double);

    // Allocate on device
    cudaMallocManaged(&a, matSize);
    cudaMallocManaged(&b, matSize);
    cudaMallocManaged(&res, matSize);

    // Copy from host to device
    cublasSetVector(N, sizeof(double), matA, 1, a, 1);
    cublasSetVector(N, sizeof(double), matB, 1, b, 1);

    // Create cublas handler
    cublasHandle_t h;
    cublasCreate(&h);

    // Carry out addition
    const double scale = 1;
    cublasDaxpy(h, N, &scale, a,1, b, 1);
    // Copy vector off gpu
    cublasGetVector(N, sizeof(double), b, 1, res, 1);

    // Copy the results out of device memory
    cudaMemcpy(matRes, res, matSize, cudaMemcpyDeviceToHost);

    // Destroy cublas
    cublasDestroy(h);

    // Free device memory
    cudaFree(a);
    cudaFree(b);
}

// Device function
__global__ void CuMatAddMatrixIKernel(long* A, long* B, long * res, long i, long j){
    long N = i * j; // Treat matrix add as vector add (same thing for equal sizes)
    long index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < N){
        res[index] = A[index] + B[index];
    }
}


void CuMatAddMatrixI(long * matA, long * matB, long * matRes, long i, long j){
    long* d_A; long *d_B; long * d_Res;
    size_t size = i*j*sizeof(long);
    // Allocate memory for CUDA
    cudaMallocManaged(&d_A, size);
    cudaMallocManaged(&d_B, size);
    cudaMallocManaged(&d_Res, size);

    // Copy over the matricies into device memory
    cudaMemcpy(d_A, matA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, matB, size, cudaMemcpyHostToDevice);

    // Set the number of threads per block and grid size
    int threadsPerBlock = 256;
    int blocksPerGrid = ((i*j) + threadsPerBlock -1) / threadsPerBlock;

    // Call the kernel
    CuMatAddMatrixIKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Res, i, j);

    // Copy the results out of device memory
    cudaMemcpy(matRes, d_Res, size, cudaMemcpyDeviceToHost);

    // Free up cuda malloc
    cudaFree(&d_A);
    cudaFree(&d_B);
    cudaFree(&d_Res);
}