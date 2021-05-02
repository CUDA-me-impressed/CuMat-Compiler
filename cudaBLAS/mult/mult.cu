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
#define BLOCK_SIZE 32 // nvidia GPUs typically have 1024 threads per block, 32*32

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}

__global__ void CuMatMatMultKernelI(const long *matA, const long *matB, long* matRes, int width, int i, int j)
{
    // Get out the indicies for the multiplication
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    // check boundry conditions incase block size is at the end
    if( r < i && c < j){
        long value = 0;
        for(int k = 0; k < width; k++){
            value += matA[r * width + k] * matB[k * j + c];
        }
        matRes[r * j + c] = value;
    }
}

extern "C" void CuMatMatMultMatrixI(HeaderI* matHeaderA, HeaderI* matHeaderB, HeaderI* matHeaderRes, long i, long p, long j){
    long* matA;
    long* matB;
    long* matRes;

    matA = matHeaderA->data;
    matB = matHeaderB->data;
    matRes = matHeaderRes->data;

    long aRank = matHeaderA->rank;
    long bRank = matHeaderB->rank;
    if(aRank == 2 && bRank == 2) {
        auto matASize = sizeof(long) * i * p;
        auto matBSize = sizeof(long) * p * j;
        auto matResSize = i * j * sizeof(long);

        cudaStream_t stream;

        // Allocate device memory
        long *d_A, *d_B, *d_Res;

        if (matRes == NULL) {
            exit(EXIT_FAILURE);
        }

        cudaMallocManaged(&d_A, matASize);
        cudaMallocManaged(&d_B, matBSize);
        cudaMallocManaged(&d_Res, matResSize);

        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

        // copy host memory to device
        cudaMemcpy(d_A, matA, matASize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, matB, matBSize, cudaMemcpyHostToDevice);

        // Setup execution parameters
        dim3 dim_grid(ceilf(i / (float)BLOCK_SIZE), ceilf(j / (float)BLOCK_SIZE), 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

        CuMatMatMultKernelI<<<dim_grid, dim_block>>>(d_A, d_B, d_Res, p, i, j);

        // Copy result from device to host
        cudaMemcpyAsync(matRes, d_Res, matResSize, cudaMemcpyDeviceToHost, stream);
        // Copy the results out of device memory
        //    cudaMemcpy(matRes, d_Res, matResSize, cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(stream);

        // Clean up memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_Res);
    }else if(aRank == 2 && bRank == 1){
        // This is Matrix-Vector product

    }else if(aRank == 1 && bRank == 1){
        // This should be Vector-Vector dot product
    }
}

// matRes(m,n) = matA(m,k) * matB(k,n)
extern "C" void CuMatMatMultMatrixD(HeaderD* matHeaderA, HeaderD* matHeaderB, HeaderD* matHeaderRes, const int m, const int k, const int n) {
    double* matA;
    double* matB;
    double* matRes;
    matA = matHeaderA->data;
    matB = matHeaderB->data;
    matRes = matHeaderRes->data;
    long aRank = matHeaderA->rank;
    long bRank = matHeaderB->rank;

    // Declare matA, matB, matRes on device
    double* d_A;
    double* d_B;
    double* d_Res;

    if(aRank == 2 && bRank == 2) {
        size_t matASize = m * k * sizeof(double);
        size_t matBSize = k * n * sizeof(double);
        size_t matResSize = m * n * sizeof(double);

        // Allocate memory for device
        cudaMallocManaged(&d_A, matASize);
        cudaMallocManaged(&d_B, matBSize);
        cudaMallocManaged(&d_Res, matResSize);

        // Copy over matA & matB to device
        cudaMemcpy(d_A, matA, matASize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, matB, matASize, cudaMemcpyHostToDevice);

        int lda = m, ldb = k, ldc = m;
        // Create a handle for CUBLAS
        cublasHandle_t handle;
        cublasCreate(&handle);
        const double alpha = 1.0f;
        const double beta = 0.0f;

//        // Timing
//        cudaEvent_t start, stop;
//        cudaEventCreate(&start);
//        cudaEventCreate(&stop);
//
//        cudaEventRecord(start);
        // Do the actual multiplication
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_B, lda, d_A, ldb, &beta, d_Res, ldc);

//        cudaEventRecord(stop);
//
//        cudaEventSynchronize(stop);
//        float milliseconds = 0;
//        cudaEventElapsedTime(&milliseconds, start, stop);
//
//        printf("The elapsed time in gpu was %.5f ms\n", milliseconds);

        // Synchronise before copy
        cudaDeviceSynchronize();

        // Copy device memory to host
        cudaMemcpy(matRes, d_Res, matResSize, cudaMemcpyDeviceToHost);

        // Destroy the handle
        cublasDestroy(handle);
    }else if(aRank == 2 && bRank == 1){
        // This is Matrix-Vector product
        size_t matASize = m * k * sizeof(double);
        size_t matBSize = k * sizeof(double);
        size_t matResSize = k * sizeof(double);

        // Allocate memory for device
        cudaMallocManaged(&d_A, matASize);
        cudaMallocManaged(&d_B, matBSize);
        cudaMallocManaged(&d_Res, matResSize);

        // Copy over matA & matB to device
        cudaMemcpy(d_A, matA, matASize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, matB, matBSize, cudaMemcpyHostToDevice);

        // Create a handle for CUBLAS
        cublasHandle_t handle;
        cublasCreate(&handle);
        const double alpha = 1.0f;
        const double beta = 0.0f;

        cublasDgemv(handle, CUBLAS_OP_N, m, k, &alpha, d_A, 2, d_B, 1, &beta, d_Res, 1);

        cudaDeviceSynchronize();

        // Copy device memory to host
        cudaMemcpy(matRes, d_Res, matResSize, cudaMemcpyDeviceToHost);

        cublasDestroy(handle);
    }
}