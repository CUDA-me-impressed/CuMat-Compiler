#include <cstring>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
/**
 * Allocates a pinned CUDA int64 matrix
 * @param hostMat
 * @param len
 * @param deviceMat
 */
//void CuMatCUDAAllocI(long * hostMat, long len, long * pinnedMat){
//    cudaError_t allocStatus = cudaMallocHost((void**)&pinnedMat, len*sizeof(long));
//    if (allocStatus != cudaSuccess)
//        printf("Error allocating pinned host memory for matrix\n");
//    memcpy(pinnedMat, hostMat, len);    // Copy the pageable matrix to a pinned matrix address
//}
//
///**
// * Allocates a pinned CUDA float64 matrix
// * @param hostMat
// * @param len
// * @param deviceMat
// */
//void CuMatCUDAAllocF(float * hostMat, long len, float * pinnedMat){
//    cudaError_t allocStatus = cudaMallocHost((void**)&pinnedMat, len*sizeof(double));
//    if (allocStatus != cudaSuccess)
//        printf("Error allocating pinned host memory for matrix\n");
//    memcpy(pinnedMat, hostMat, len);    // Copy the pageable matrix to a pinned matrix address
//}
//
//void CuMatCUDAFreeI(long *pinnedMat){
//    cudaFreeHost(pinnedMat);
//}
//
//void CuMatCUDAFreeF(double *pinnedMat){
//    cudaFreeHost(pinnedMat);
//}