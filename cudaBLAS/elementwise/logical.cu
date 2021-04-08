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

// Device function
__global__ void CuMatLORMatrixDKernel(double* A, double* B, double * res, long i, long j){
    long N = i * j; // Treat matrix add as vector add (same thing for equal sizes)
    long index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < N){
        res[index] = (double) (A[index] || B[index]);
    }
}

extern "C" void CuMatLORMatrixD(double * matA, double * matB, double * matRes, long i, long j){
    double* d_A; double *d_B; double * d_Res;
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
    CuMatLORMatrixDKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Res, i, j);

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
__global__ void CuMatLORMatrixIKernel(long* A, long* B, long * res, long i, long j){
    long N = i * j; // Treat matrix add as vector add (same thing for equal sizes)
    long index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < N){
        res[index] = (long) (A[index] || B[index]);
    }
}


extern "C" void CuMatLORMatrixI(long * matA, long * matB, long * matRes, long i, long j){
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
    CuMatLORMatrixIKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Res, i, j);

    // Synchronise before copying
    cudaDeviceSynchronize();

    // Copy the results out of device memory
    cudaMemcpy(matRes, d_Res, size, cudaMemcpyDeviceToHost);

    // Free up cuda malloc
    cudaFree(&d_A);
    cudaFree(&d_B);
    cudaFree(&d_Res);
}


/*
 * Logical AND Kernel functions
 */


// Device function
__global__ void CuMatLANDMatrixDKernel(double* A, double* B, double * res, long i, long j){
    long N = i * j; // Treat matrix add as vector add (same thing for equal sizes)
    long index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < N){
        res[index] = (double) (A[index] && B[index]);
    }
}

extern "C" void CuMatLANDMatrixD(double * matA, double * matB, double * matRes, long i, long j){
    double* d_A; double *d_B; double * d_Res;
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
    CuMatLANDMatrixDKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Res, i, j);

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
__global__ void CuMatLANDMatrixIKernel(long* A, long* B, long * res, long i, long j){
    long N = i * j; // Treat matrix add as vector add (same thing for equal sizes)
    long index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < N){
        res[index] = (long) (A[index] && B[index]);
    }
}


extern "C" void CuMatLANDMatrixI(long * matA, long * matB, long * matRes, long i, long j){
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
    CuMatLANDMatrixIKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Res, i, j);

    // Synchronise before copying
    cudaDeviceSynchronize();

    // Copy the results out of device memory
    cudaMemcpy(matRes, d_Res, size, cudaMemcpyDeviceToHost);

    // Free up cuda malloc
    cudaFree(&d_A);
    cudaFree(&d_B);
    cudaFree(&d_Res);
}
