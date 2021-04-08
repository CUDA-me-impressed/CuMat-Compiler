#include "elementwise/add.cu"
#include "elementwise/sub.cu"
#include "elementwise/logical.cu"
#include "elementwise/div.cu"

#include "mult/mult.cu"
#include "utils.cu"
#include <iostream>

int main(){
    double A[6] = {1,0,7,3,4,1};
    double B[6] = {4,3,0,2,1,3};
    double C[9];

    CuMatDivMatrixD(A,B,C,6);
    for(int i = 0; i < 6; i++){
        std::cout << C[i] << std::endl;
    }
}