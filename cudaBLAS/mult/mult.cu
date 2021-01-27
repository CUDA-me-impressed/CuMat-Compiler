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
    float *d_A, *d_B, *d_Res;


    // Allocate host matrix C
    float *h_C;
    cudaMallocHost(&h_Res, matResSize);

    if (h_C == NULL) {
        exit(EXIT_FAILURE);
    }

    cudaMalloc(reinterpret_cast<void **>(&d_A), matASize);
    cudaMalloc(reinterpret_cast<void **>(&d_B), matBSize);
    cudaMalloc(reinterpret_cast<void **>(&d_C), matResSize);

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // copy host memory to device
    cudaMemcpyAsync(d_A, h_A, matASize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, matBSize, cudaMemcpyHostToDevice, stream);

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(p / threads.x, p / threads.y);

    // Performs warmup operation using matrixMul CUDA kernel
    if (block_size == 16) {
        MatrixMulCUDA<16>
            <<<grid, threads, 0, stream>>>(d_Res, d_A, d_B, i, p);
    } else {
        MatrixMulCUDA<32>
            <<<grid, threads, 0, stream>>>(d_Res, d_A, d_B, i, p);
    }

    cudaStreamSynchronize(stream);

    // Execute the kernel
    int nIter = 300;

    for (int itt = 0; itt < nIter; itt++) {
        if (block_size == 16) {
        MatrixMulCUDA<16>
            <<<grid, threads, 0, stream>>>(d_Res, d_A, d_B, i, p);
        } else {
        MatrixMulCUDA<32>
            <<<grid, threads, 0, stream>>>(d_Res, d_A, d_B, i, p);
        }
    }

    // Copy result from device to host
    cudaMemcpyAsync(h_Res, d_Res, matResSize, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Clean up memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}