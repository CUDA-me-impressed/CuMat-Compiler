./cmake-build-release/src/compiler/src/CuMatComp "$@"
retVal=$?
if [ $retVal -ne 0 ]; then
  echo "Error on CuMat Compiler: " $retVal
  exit $retVal

clang++-10 -c ./$1.ll
retVal=$?
if [ $retVal -ne 0 ]; then
  echo "Error on Clang Compiler: " $retVal
  exit $retVal

nvcc -g -lcublas -o ./$1-Program ./$1.o ./cudaBLAS/elementwise/*.cu ./cudaBLAS/utils/*.cpp ./cudaBLAS/mult/*.cu
retVal=$?
if [ $retVal -ne 0 ]; then
  echo "Error on NVCC Compiler: " $retVal
  exit $retVal

chmod +x $1-Program
retVal=$?
if [ $retVal -ne 0 ]; then
  echo "Error on chmod: " $retVal
  exit $retVal

rm ./$1.o ./$1.
retVal=$?
if [ $retVal -ne 0 ]; then
  echo "Error on rm: " $retVal
  exit $retVal
