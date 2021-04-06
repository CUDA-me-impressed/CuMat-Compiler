#include "addition/add.cu"
#include "mult/mult.cu"
#include <iostream>

int main(){
    long A[6] = {1,2,7,3,4,1};
    long B[6] = {4,3,8,2,1,3};
    long C[9];

    CuMatMultMatrixI(A,B,C,3,2,3);
    for(int i = 0; i < 9; i++){
        std::cout << C[i] << std::endl;
    }
}