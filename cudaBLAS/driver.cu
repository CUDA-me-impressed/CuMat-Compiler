#include "elementwise/add.cu"
#include "elementwise/sub.cu"
#include "elementwise/logical.cu"
#include "elementwise/div.cu"
#include "elementwise/mult.cu"

#include "mult/mult.cu"
#include "utils.cu"
#include "utils/io.cpp"
#include <iostream>


extern void CuMatAddMatrixI(long*,long*,long*,long);
extern void printMatrix(long*,long);

int main(){
    long A[6] = {1,2,7,3};
    long B[6] = {1,0,0,1};
    long C[6];

    CuMatAddMatrixI(A,B,C,4);

    printMatrixI(C,4);
    //for(int i = 0; i < 6; i++){
    //    std::cout << C[i] << std::endl;
    //}
}
