#include "addition/add.cu"
#include "mult/mult.cu"
#include <iostream>

int main(){
    double A[4] = {1,2,3,4};
    double B[4] = {4,3,2,1};
    double C[4];

    CuMatMultMatrixD(A,B,C,2,2,2);
    for(int i = 0; i < 4; i++){
        std::cout << C[i] << std::endl;
    }
}