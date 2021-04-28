import time
import os

# start = time.perf_counter()
# os.system("./CuMat-testBasic.cmProgram")
# end = time.perf_counter()
# print(end-start)
files = [2,4,6,8,10,20,40,60,80,100,150,200,250,300,400,500,750,1000]
for i in files:
    filename = "./precompiled-cpu/" + str(i) + ".cmProgram"
    start = time.perf_counter()
    os.system(filename)
    end = time.perf_counter()
    timetaken = (end-start) * 1000
    print(f"Took: {timetaken}")