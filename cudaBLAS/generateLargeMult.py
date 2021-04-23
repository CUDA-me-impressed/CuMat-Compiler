import numpy as np

def printAsCuMatMatrix(mat):
    np.set_printoptions(threshold=9999999999)
    file = "["
    file = file + np.array2string(mat, separator=', ').replace('],', '\\').replace('[', '').replace(']','')
    file = file + ']'
    return file

n = 257
dim = (n,n)

x = np.ones(dim) * 2.1
# y = np.identity(dim[0]) * 1

programHeader = "func int main { a = "
programHeader = programHeader + printAsCuMatMatrix(x).replace("\n", "")
programHeader = programHeader + "\n"
programHeader = programHeader + "return a .* a"
programHeader = programHeader + "}"
text_file = open("pythonoutput.cm", "w")

text_file.write(programHeader)

text_file.close()