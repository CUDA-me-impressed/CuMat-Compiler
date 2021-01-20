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

    // Res needs to be allocated on the host
    res = (double*) malloc(matSize);
    // Allocate on device
    cudaMallocManaged(&a, matSize);
    cudaMallocManaged(&b, matSize);

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

    // Destroy cublas
    cublasDestroy(h);
}