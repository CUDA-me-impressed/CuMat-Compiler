import os

for n in [2,4,6,8,10,20,40,60,80,100,150,200,250,300,400,500,750,1000]:
    filename = str(n) + ".cm"
    os.system("./compile.sh " + filename)
