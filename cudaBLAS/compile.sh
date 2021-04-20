clang++-10 -c ./$1.ll -fPIE
nvcc -g -lcublas -o ./$1Program ./$1.o elementwise/*.cu utils/*.cpp mult/*.cu
chmod +x $1Program
rm $1.o
