import numpy as np
import sys, os
sys.path.append(os.pardir)
from scratchnet.links import *

conv_fast = Convolution2DFast(3, 16, pad=1)

x = np.random.rand(2, 3, 64, 64) #(batch, channel, H, W)
y = conv_fast.forward(x)
print(y.shape)

dy = np.zeros_like(y.shape)
dx = conv_fast.backward(y)
print(dx.shape)
print(conv_fast.dW.shape)
