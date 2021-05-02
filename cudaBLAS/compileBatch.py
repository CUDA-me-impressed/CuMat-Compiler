import os

for n in [300,400,500,750,1000]:
    filename = str(n) + ".cm"
    os.system("./compile.sh " + filename)
