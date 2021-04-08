#include "elementwise/add.cu"
#include <iostream>

int main(){
    long A[6] = {1,2,7,3,4,1};
    long B[6] = {4,3,8,2,1,3};
    long C[6] = {0,0,0,0,0,0};

    CuMatAddMatrixI(A,B,C,2,3);
    for(int i = 0; i < 6; i++){
        std::cout << C[i] << std::endl;
    }
}
