import sys, os
sys.path.append(os.pardir)
from scratchnet.functions import *
from scratchnet.lib.utils import * 
import numpy as np

class LocallyConnectedLayer:
    def __init__(self, in_channel, out_channel, in_h, in_w, ksize=3, stride=1, pad=0):
        self.pad = pad
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.ksize = ksize
        self.in_h = in_h
        self.in_w = in_w
        self.out_h = int((in_h + 2*self.pad - self.ksize) / self.stride) + 1
        self.out_w = int((in_w + 2*self.pad - self.ksize) / self.stride) + 1
        self.kernel = np.random.randn(out_channel, in_channel, self.out_h, self.out_w, ksize, ksize) * 0.01

        self.bias = np.zeros((out_channel, self.out_h, self.out_w))

    def forward(self, x):
        batch_size = x.shape[0]

        self.x = x
        self.input_col = im2col(x, self.ksize, self.ksize, self.stride, self.pad) # (batch_size, in_channel, in_h, in_w) -> (batch_size * out_h * out_w, in_channel * ksize * ksize) 
        self.weight_col = self.kernel.transpose(0, 2, 3, 1, 4, 5) # (out_channel, in_channel, out_h, out_w, ksize, ksize) -> (out_channel, out_h, out_w, in_channel * ksize, ksize)
        self.weight_col = self.weight_col.reshape(self.out_channel, self.out_h * self.out_w, self.in_channel * self.ksize * self.ksize) # (out_channel, out_h * out_w, in_channel * ksize * ksize)
        self.weight_col = np.tile(self.weight_col, (1, batch_size, 1)) # batch_size分だけ第2次元をrepeate (out_channel, batch_size * out_h * out_w, in_channel * ksize * ksize)
        out = np.sum(self.input_col * self.weight_col, axis=2) # sumを取って、(out_channel, batch_size * out_h * out_w)に変形
        out += np.tile(self.bias.reshape(self.out_channel, -1), (1, batch_size)) # (out_channel, out_h, out_w) -> (out_channel, out_h * out_w) -> (out_channel, batch_size * out_h * out_w)
        out = out.reshape(self.out_channel, batch_size, self.out_h, self.out_w).transpose(1, 0, 2, 3) # (batch_size, out_channel, out_h, out_w)

        return out

    def backward(self, dout):
        batch_size = dout.shape[0]
        self.db = np.sum(dout, axis=0) # (batch_size, out_channel, out_h, out_w)のbatch_size分全て合計してbの勾配を計算
        dout = dout.transpose(1, 0, 2, 3) # (batch_size, out_channel, out_h, out_w) -> (out_channel, batch_size, out_h, out_w)
        dout = dout.reshape(self.out_channel, -1) # (out_channel, batch_size, out_h, out_w) -> (out_channel, batch_size * out_h * out_w)
        dout = np.broadcast_to(dout[:, :, np.newaxis], (self.out_channel, batch_size * self.out_h * self.out_w, self.in_channel * self.ksize * self.ksize))
        self.dW = self.input_col * dout # (out_channel * batch_size * out_h * out_w, self.in_channel * ksize * ksize) x (batch_size * out_h * out_w, self.in_channel * ksize * ksize) 
        self.dW = np.sum(self.dW.reshape(self.out_channel, batch_size, self.out_h * self.out_w, self.in_channel * self.ksize * self.ksize), axis=1)
        self.dW = self.dW.reshape(self.out_channel, self.out_h, self.out_w, self.in_channel, self.ksize, self.ksize).transpose(0, 3, 1, 2, 4, 5)
 
        dcol = np.sum((dout * self.weight_col), axis=0)
        dx = col2im(dcol, self.x.shape, self.ksize, self.ksize, self.stride, self.pad)

        return dx

    def numerical_grad(self, x): 
        loss_W = lambda W: self.forward(x)

        grads = {}
        grads['dW'] = numerical_gradient(loss_W, self.kernel)
        grads['db'] = numerical_gradient(loss_W, self.bias)

        return grads


