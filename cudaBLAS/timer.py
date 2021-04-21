import time
import os

start = time.perf_counter()
os.system("./CuMat-testBasic.cmProgram")
end = time.perf_counter()
print(end-start)