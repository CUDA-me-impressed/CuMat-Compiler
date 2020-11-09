#include "ParMat.hpp"
#include <math.h>

template<unsigned n, unsigned m> Optimisations::Matrix<n,m>::Matrix()
{
    // We initially create the identity matrix
    for(int j = 0; j < m; j++){
        for(int i = 0; i < n; i++){
            data[m][n] = m == n ? 1 : 0;
        }
    }
}

template<unsigned int n, unsigned int m>
template<unsigned int x, unsigned int y>
void Optimisations::Matrix<n, m>::copy(int startX, int endX, int startY, int endY, Optimisations::Matrix<x, y> &out) {
    static_assert((endX-startX == x) && (endY-startY == y), "Matrices template parameters do not match");
    for(int j = startY; j < endY; j++){
        for(int i = startX; i < endX; i++){
            out.data[i-startX][j-startY] = data[i][j];
        }
    }
}

template<unsigned int n, unsigned int m>
void Optimisations::Matrix<n, m>::add(const Optimisations::Matrix<n, m> &m2,
                                      Optimisations::Matrix<n, m> &out) {
    // Divide and conquer algorithm -> Nice and simple
    const int hozSeg = std::floor(n / MINSIZE);
    const int verSeg = std::floor(m / MINSIZE);
}


