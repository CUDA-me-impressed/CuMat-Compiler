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

#define block_size 16


template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(long *C, long *A, long *B, int wA, int wB) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    long Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ long As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ long Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}



void CuMatMultMatrixI(long * matA, long * matB, long * matRes, long i, long p, long j){
    auto matASize = sizeof(long) * i * p;
    auto matBSize = sizeof(long) * p * j;
    auto matResSize = i * j * sizeof(long);

    long* h_A; long * h_B; long *h_Res;

    // Allocate cuda managed host memory
    cudaMallocManaged(&h_A, matASize);
    cudaMallocManaged(&h_B, matBSize);
    cudaMallocManaged(&h_Res, matResSize);

    // Copy over the data from the function pointers
    cudaMemcpy(h_A, matA, matASize, cudaMemcpyHostToDevice);
    cudaMemcpy(h_B, matB, matBSize, cudaMemcpyHostToDevice);

    cudaStream_t stream;

    // Allocate device memory
    long *d_A, *d_B, *d_Res;

    if (h_Res == NULL) {
        exit(EXIT_FAILURE);
    }

    cudaMalloc(reinterpret_cast<void **>(&d_A), matASize);
    cudaMalloc(reinterpret_cast<void **>(&d_B), matBSize);
    cudaMalloc(reinterpret_cast<void **>(&d_Res), matResSize);

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // copy host memory to device
    cudaMemcpyAsync(d_A, h_A, matASize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, matBSize, cudaMemcpyHostToDevice, stream);

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(p / threads.x, p / threads.y);



    cudaStreamSynchronize(stream);

    if (block_size == 16) {
        MatrixMulCUDA<16>
            <<<grid, threads, 0, stream>>>(d_Res, d_A, d_B, i, p);
    } else {
        MatrixMulCUDA<32>
            <<<grid, threads, 0, stream>>>(d_Res, d_A, d_B, i, p);
    }


    // Copy result from device to host
    cudaMemcpyAsync(h_Res, d_Res, matResSize, cudaMemcpyDeviceToHost, stream);
    // Copy the results out of device memory
    cudaMemcpy(matRes, d_Res, matResSize, cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);

    // Clean up memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_Res);
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
