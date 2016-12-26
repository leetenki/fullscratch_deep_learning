import numpy as np

# yとtはnumpyの1次元配列
# t: one-hot表現[0, 0, 1, 0]
# y: 確率表現[0.1, 0.2, 0.3, 0.4]
# return number
def cross_entropy_error(y, t): 
    delta = 1e-7

    if y.ndim == 1:
        t.reshape(1, t.size)
        y.reshape(1, y.size)

    if y.size == t.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
