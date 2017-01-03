import numpy as np
import sys, os
sys.path.append(os.pardir)
from scratchnet.links import *

local_layer = LocallyConnectedLayer(3, 8, 5, 5) # (in_ch, out_ch, in_h, out_w)

x = np.random.rand(2, 3, 5, 5) #(batch, channel, H, W)
y = local_layer.forward(x)
dy = np.ones(y.shape)
dx = local_layer.backward(dy)

numerical_grad = local_layer.numerical_grad(x)
print(np.sum(numerical_grad["dW"] - local_layer.dW))
print(np.sum(numerical_grad["db"] - local_layer.db))

print(local_layer.dW.shape)
print(dx.shape)
