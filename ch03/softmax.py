import numpy as np

def softmax(x):
    x -= np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

x = np.array([100000, 100005, 100003])
print(softmax(x))
