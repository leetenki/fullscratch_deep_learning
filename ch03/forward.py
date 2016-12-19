import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

X = np.array([1, 2])
W1 = np.array([[1, 2], [3, 4], [5, 6]])
b1 = np.array([0.1, 0.2, 0.3])

W2 = np.array([[1, 2, 3]])
b2 = np.array([0.1])

h = np.dot(X, W1.T) + b1
h = sigmoid(h)
h = np.dot(h, W2.T) + b2
Y = identity_function(h)

print(Y)
