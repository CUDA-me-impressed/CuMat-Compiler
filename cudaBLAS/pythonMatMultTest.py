import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import numpy as np
import timeit

def printAsCuMatMatrix(mat):
    np.set_printoptions(threshold=9999999999)
    file = "["
    file = file + np.array2string(mat, separator=', ').replace('],', '\\').replace('[', '').replace(']','')
    file = file + ']'
    return file

sizes = [100,200,300,400,500,600,700,800,900,1000, 2000,3000,4000,5000,6000,7000,8000,9000]

for n in sizes:
    dim = (n,n)

    x = np.random.rand(n, n)

    time = timeit.timeit(
    lambda: np.matmul(x,x),
    number=1
    )
    print(f"{time} s".format(time * 1000))