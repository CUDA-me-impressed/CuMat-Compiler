/*
 * Generalised Matrix Vector multiplication cuda functions
 */

#include <cstring>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "../utils/headers.hpp"

__global__ void CuMatGEMVKernelI(long* vec, long* mat, long* res, long N, long M) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    long sum = 0;
    if (tid < M) {
        for (int i = 0; i < N; i++) sum += vec[i] * mat[(i * M) + tid];
        res[tid] = sum;
    }
}

extern "C" void CuMatGEMVMatrixI(HeaderI* matHeaderA, HeaderI* matHeaderB, HeaderI* matHeaderRes, long N, long M) {
    long* vecA;
    long* matB;
    long* vecRes;

    vecA = matHeaderA->data;
    matB = matHeaderB->data;
    vecRes = matHeaderRes->data;

    long *d_A, *d_B, *d_Res;

    cudaMalloc((void**)&d_A, sizeof(long) * N);
    cudaMalloc((void**)&d_B, sizeof(long) * N * M);
    cudaMalloc((void**)&d_Res, sizeof(long) * M);

    cudaMemcpy(d_A, vecA, sizeof(long) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, matB, sizeof(long) * N * M, cudaMemcpyHostToDevice);

    // Takes in Vector, Matrix, Resultant Vector and sizes
    CuMatGEMVKernelI<<<M / 256 + 1, 256>>>(d_A, d_B, d_Res, N, M);

    cudaMemcpy(vecRes, d_Res, sizeof(long) * M, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_Res);
}