from scipy.optimize import curve_fit
import numpy as np

xdata =  [70,80,90,100,110,120,130,140,150,160,
170,180,190,200,250,300,400,500,750,1000]

ydata = [0.163639,0.230039,0.286963,0.343532,0.486311,0.511364,0.460507,0.604441,0.7165,0.789863,0.900274,0.968071,0.970639,1.072578,1.356337,1.740857,1.988335,2.903016,3.637857,4.235182]


def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)

p0 = [max(ydata), np.median(xdata),1,min(ydata)] # this is an mandatory initial guess

popt, pcov = curve_fit(sigmoid, xdata, ydata)

print(popt)
