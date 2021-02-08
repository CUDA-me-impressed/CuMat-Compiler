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

#define BLOCK_SIZE 32 // nvidia GPUs typically have 1024 threads per block, 32*32

__global__ void CuMatMatMultKernel(const long *matA, const long *matB, long* matRes, int width, int i, int j)
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

void CuMatMultMatrixI(long * matA, long * matB, long * matRes, long i, long p, long j){
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
    dim3 dim_grid(ceilf(i/(float)BLOCK_SIZE), ceilf(j/(float)BLOCK_SIZE), 1);
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);


    CuMatMatMultKernel<<<dim_grid, dim_block>>>(d_A, d_B, d_Res, p,  i, j);

    // Copy result from device to host
    cudaMemcpyAsync(matRes, d_Res, matResSize, cudaMemcpyDeviceToHost, stream);
    // Copy the results out of device memory
    cudaMemcpy(matRes, d_Res, matResSize, cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);

    // Clean up memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_Res);
}

// matRes(m,n) = matA(m,k) * matB(k,n)
void CuMatMultMatrixD(const double *matA, const double *matB, double *matRes, const int m, const int k, const int n) {
    // Declare matA, matB, matRes on device
    double* d_A;
    double* d_B;
    double* d_Res;

    size_t matASize = m * k * sizeof(double);
    size_t matBSize = k * n * sizeof(double);
    size_t matResSize = m * n * sizeof(double);

    // Allocate memory for device
    cudaMalloc(&d_A, matASize);
    cudaMalloc(&d_B, matBSize);
    cudaMalloc(&d_Res, matResSize);

    // Copy over matA & matB to device
    cudaMemcpy(d_A, matA, matASize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, matB, matASize, cudaMemcpyHostToDevice);

    int lda=m,ldb=k,ldc=m;
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    const double alpha = 1.0f;
    const double beta = 0.0f;

    // Do the actual multiplication
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_B, lda, d_A, ldb, &beta, d_Res, ldc);

    // Synchronise before copy
    cudaDeviceSynchronize();

    // Copy device memory to host
    cudaMemcpy(matRes,d_Res,matResSize,cudaMemcpyDeviceToHost);

    // Destroy the handle
    cublasDestroy(handle);
}
