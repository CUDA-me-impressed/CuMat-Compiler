#include "elementwise/add.cu"
#include "elementwise/sub.cu"
#include "elementwise/logical.cu"
#include "elementwise/div.cu"

#include "mult/mult.cu"
#include "utils.cu"
#include "utils/io.cpp"
#include <iostream>

extern void CuMatAddMatrixI(long*,long*,long*,long);
extern void printMatrix(long*,long);

int main(){
    long A[6] = {1,2,7,3,4,1};
    long B[6] = {4,3,8,2,1,3};
    long C[6];

    CuMatAddMatrixI(A,B,C,6);

    printMatrix(C,6);
    //for(int i = 0; i < 6; i++){
    //    std::cout << C[i] << std::endl;
    //}
}
