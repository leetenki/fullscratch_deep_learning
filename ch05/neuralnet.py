import sys, os
sys.path.append(os.pardir)
from scratchnet.links import *
import numpy as np

x = np.array([[1, 1, 1], [2, 2, 2]])
t = np.array([[0, 0, 1], [1, 0, 0]])

relu = ReLU()
sigmoid = Sigmoid()
linear = Linear(3, 2)
softmax_cross_entropy = SoftmaxCrossEntropy()

print(relu.forward(x))
print(sigmoid.forward(x))
print(linear.forward(x))
print(softmax_cross_entropy.forward(x, t))
print(softmax_cross_entropy.backward())
