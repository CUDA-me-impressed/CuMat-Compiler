#include "elementwise/add.cu"
#include "mult/mult.cu"
#include "utils.cu"
#include <iostream>

int main(){
    long A[6] = {1,2,7,3,4,1};
    long B[6] = {4,3,8,2,1,3};
    long C[9];

    CuMatMatMultMatrixI(A,B,C,3,2,3);
    for(int i = 0; i < 9; i++){
        std::cout << C[i] << std::endl;
    }

    std::cout << "With pinned memory:" << std::endl;
    long * A_pinned;
    long * B_pinned;
    long * C_pinned;
    CuMatCUDAAllocI(A, 6, A_pinned);
    CuMatCUDAAllocI(B, 6, B_pinned);

    CuMatMatMultMatrixI(A_pinned,B_pinned,C,3,2,3);
    for(int i = 0; i < 9; i++){
        std::cout << C[i] << std::endl;
    }

    CuMatCUDAFreeI(A_pinned);
    CuMatCUDAFreeI(B_pinned);
    CuMatCUDAFreeI(C_pinned);
}