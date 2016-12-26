import sys, os
sys.path.append(os.pardir)
from scratchnet.functions import *
import numpy as np

class simpleNet:
    def __init__(self, n_in, n_out):
        self.W = np.random.randn(n_out, n_in)

    def predict(self, x):
        return np.dot(x, self.W.T)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

net = simpleNet(10, 2)
x = np.random.rand(10)
t = np.array([1, 0])

def f(W):
    temp_W = net.W
    net.W = W
    loss = net.loss(x, t)
    net.W = temp_W
    return loss

print(partial_numerical_grad(f, net.W))
