import numpy as np
import os

n = 1000
data = ""
for i in range (n*n):
    data = data + str(1)
    if i < ((n*n)-1):
        data = data + ","

text_file = open("data.txt", "w")

text_file.write(data)

text_file.close()