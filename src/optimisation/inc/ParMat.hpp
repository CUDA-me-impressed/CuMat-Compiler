#pragma once

namespace Optimisations {
#define MINSIZE 2

    template<unsigned n, unsigned m>
    class Matrix {
    public:
        Matrix();
        void add(const Matrix<n,m> & m2, Matrix<n,m> & out);

        template<unsigned x, unsigned y>
        void copy(int startX, int endX, int startY, int endY, Matrix<x,y> & out);

        double data[n][m];

    };
}