import numpy as np
import timeit

def printAsCuMatMatrix(mat):
    np.set_printoptions(threshold=9999999999)
    file = "["
    file = file + np.array2string(mat, separator=', ').replace('],', '\\').replace('[', '').replace(']','')
    file = file + ']'
    return file


dim = (200,200)

x = np.ones(dim, dtype=np.float64) * 2.1
y = np.ones(dim) * 2.65

time = timeit.timeit(
 lambda: np.matmul(np.matmul(x,x),y),
 number=10
)
print(f"{time} ms".format(time * 1000/ 10))