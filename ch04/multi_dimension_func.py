import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def func(x1, x2):
    return x1 ** 3 + x2 ** 3

x1 = np.arange(-10.0, 10.0, 1)
x2 = np.arange(-10.0, 10.0, 1)
X1, X2 = np.meshgrid(x1, x2)
Y = func(X1, X2)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(X1, X2, Y)
plt.show()
