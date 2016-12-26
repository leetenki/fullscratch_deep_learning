import numpy as np

# xはnumpy.array
# return値もnumpy.array
def softmax(x):
    x -= np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)
