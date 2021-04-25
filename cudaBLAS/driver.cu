#include "elementwise/add.cu"
#include "elementwise/sub.cu"
#include "elementwise/logical.cu"
#include "elementwise/div.cu"
#include "elementwise/mult.cu"

#include "mult/mult.cu"
#include "mult/gemv.cu"
#include "utils.cu"
#include "utils/io.cpp"
#include "utils/headers.hpp"
#include <iostream>


extern void CuMatAddMatrixI(long*,long*,long*,long);
extern void printMatrix(long*,long);

int main(){
    long A[4] = {1,0,0,1};
    long B[2] = {1,0};
    long C[2];

    CuMatGEMVMatrixI(A,B,C,4);

    printMatrixI(C,4);
    //for(int i = 0; i < 6; i++){
    //    std::cout << C[i] << std::endl;
    //}
}
