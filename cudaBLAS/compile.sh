clang++-10 -c $1.ll
nvcc -g -lcublas -o $1 ./$1.o elementwise/*.o utils/*.o
