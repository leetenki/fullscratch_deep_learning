import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size

y = np.array([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]])
t = np.array([[0, 0, 1], [0, 1, 0]])

print(cross_entropy_error(y, t))
