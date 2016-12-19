import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

y = np.array([0.2, 0.3, 0.5])
t = np.array([0, 0, 1])

print(cross_entropy_error(y, t))
