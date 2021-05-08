import numpy as np
import os

def printAsCuMatMatrix(mat):
    np.set_printoptions(threshold=9999999999)
    file = "["
    file = file + np.array2string(mat, separator=', ').replace('],', '\\').replace('[', '').replace(']','')
    file = file + ']'
    return file


# for n in [300,400,500,750,1000]:
n = 10
dim = (n,n)

x = np.ones(dim) * 2.1
# y = np.identity(dim[0]) * 1

programHeader = "func float[" + str(n) + "," + str(n) + "] main { a = "
programHeader = programHeader + printAsCuMatMatrix(x).replace("\n", "")
programHeader = programHeader + "\n"
programHeader = programHeader + "return a * a"
programHeader = programHeader + "}"
text_file = open("pythonoutput.cm", "w")

text_file.write(programHeader)

text_file.close()

os.system("cp pythonoutput.cm ../cmake-build-debug/src/compiler/src/python.cm")
os.system("cp pythonoutput.cm ../cmake-build-debug/src/compiler/src/python.cm")
