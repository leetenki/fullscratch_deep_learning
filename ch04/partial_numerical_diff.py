import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def partial_numerical_diff(f, x):
    h = 1e-4
    return (
        (f(x[0]+h, x[1]) - f(x[0]-h, x[1])) / (2*h),
        (f(x[0], x[1]+h) - f(x[0], x[1]-h)) / (2*h)
    )

def func(x1, x2):
    return x1 ** 3 + x2 ** 3

x1 = np.arange(-10.0, 10.0, 1)
x2 = np.arange(-10.0, 10.0, 1)
X1, X2 = np.meshgrid(x1, x2)
Y = func(X1, X2)

for x in x1:
    for y in x2:
        print(partial_numerical_diff(func, np.array([x, y])))
