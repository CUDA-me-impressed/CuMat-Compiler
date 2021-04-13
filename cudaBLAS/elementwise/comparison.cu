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
__global__ void CuMatLTMatrixDKernel(double* A, double* B, double * res, long len){
    long index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < len){
        res[index] = static_cast<double>(A[index] < B[index]);
    }
}

extern "C" void CuMatLTMatrixD(double * matA, double * matB, double * matRes, long len){
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
    CuMatLTMatrixDKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Res, len);

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
__global__ void CuMatLTMatrixIKernel(long* A, long* B, long * res, long len){
    long index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < len){
        res[index] = static_cast<long>(A[index] < B[index]);
    }
}


extern "C" void CuMatLTMatrixI(long * matA, long * matB, long * matRes, long len){
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
    CuMatLTMatrixIKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Res, len);

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
__global__ void CuMatGTMatrixDKernel(double* A, double* B, double * res, long len){
    long index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < len){
        res[index] = static_cast<double>(A[index] > B[index]);
    }
}

extern "C" void CuMatGTMatrixD(double * matA, double * matB, double * matRes, long len){
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
    CuMatGTMatrixDKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Res, len);

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
__global__ void CuMatGTMatrixIKernel(long* A, long* B, long * res, long len){
    long index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < len){
        res[index] = static_cast<long>(A[index] > B[index]);
    }
}


extern "C" void CuMatGTMatrixI(long * matA, long * matB, long * matRes, long len){
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
    CuMatGTMatrixIKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Res, len);

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
__global__ void CuMatLTEMatrixDKernel(double* A, double* B, double * res, long len){
    long index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < len){
        res[index] = static_cast<double>(A[index] <= B[index]);
    }
}

extern "C" void CuMatLTEMatrixD(double * matA, double * matB, double * matRes, long len){
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
    CuMatLTEMatrixDKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Res, len);

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
__global__ void CuMatLTEMatrixIKernel(long* A, long* B, long * res, long len){
    long index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < len){
        res[index] = static_cast<long>(A[index] <= B[index]);
    }
}


extern "C" void CuMatLTEMatrixI(long * matA, long * matB, long * matRes, long len){
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
    CuMatLTEMatrixIKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Res, len);

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
__global__ void CuMatGTEMatrixDKernel(double* A, double* B, double * res, long len){
    long index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < len){
        res[index] = static_cast<double>(A[index] >= B[index]);
    }
}

extern "C" void CuMatGTEMatrixD(double * matA, double * matB, double * matRes, long len){
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
    CuMatGTEMatrixDKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Res, len);

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
__global__ void CuMatGTEMatrixIKernel(long* A, long* B, long * res, long len){
    long index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < len){
        res[index] = static_cast<long>(A[index] >= B[index]);
    }
}


extern "C" void CuMatGTEMatrixI(long * matA, long * matB, long * matRes, long len){
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
    CuMatGTEMatrixIKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Res, len);

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
__global__ void CuMatEQMatrixDKernel(double* A, double* B, double * res, long len){
    long index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < len){
        res[index] = static_cast<double>(A[index] == B[index]);
    }
}

extern "C" void CuMatEQMatrixD(double * matA, double * matB, double * matRes, long len){
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
    CuMatEQMatrixDKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Res, len);

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
__global__ void CuMatEQMatrixIKernel(long* A, long* B, long * res, long len){
    long index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < len){
        res[index] = static_cast<long>(A[index] == B[index]);
    }
}


extern "C" void CuMatEQMatrixI(long * matA, long * matB, long * matRes, long len){
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
    CuMatEQMatrixIKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Res, len);

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
__global__ void CuMatNEQMatrixDKernel(double* A, double* B, double * res, long len){
    long index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < len){
        res[index] = static_cast<double>(A[index] != B[index]);
    }
}

extern "C" void CuMatNEQMatrixD(double * matA, double * matB, double * matRes, long len){
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
    CuMatNEQMatrixDKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Res, len);

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
__global__ void CuMatNEQMatrixIKernel(long* A, long* B, long * res, long len){
    long index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < len){
        res[index] = static_cast<long>(A[index] != B[index]);
    }
}


extern "C" void CuMatNEQMatrixI(long * matA, long * matB, long * matRes, long len){
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
    CuMatNEQMatrixIKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Res, len);

    // Synchronise before copying
    cudaDeviceSynchronize();

    // Copy the results out of device memory
    cudaMemcpy(matRes, d_Res, size, cudaMemcpyDeviceToHost);

    // Free up cuda malloc
    cudaFree(&d_A);
    cudaFree(&d_B);
    cudaFree(&d_Res);
}