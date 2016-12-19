import numpy as np

def mean_squared_error(y, t):
    return np.sum((y-t)**2) / 2

t = np.array([1, 0, 0])
y = np.array([0.2, 0.3, 0.5])

print(mean_squared_error(y, t))
