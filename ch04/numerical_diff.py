import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def func(x):
    return 0.01 * x ** 2 + 0.1 * x

x = np.arange(0.0, 20.0, 0.1)
y = func(x)
y_grad = numerical_diff(func, x)

plt.plot(x, y)
plt.plot(x, y_grad)
plt.show()
