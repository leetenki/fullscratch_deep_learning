import sys, os
sys.path.append(os.pardir)
from scratchnet.lib.utils import * 
import numpy as np

class Convolution2D:
    def __init__(self, in_channel, out_channel, ksize=3, stride=2, pad=0):
        self.pad = pad
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.ksize = ksize
        self.kernel = np.random.randn(out_channel, in_channel, ksize, ksize)
        self.bias = np.zeros((out_channel, 1, 1))

    def forward(self, x):
        x = np.pad(x, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), "constant")

        out_h = int((x.shape[2] - self.ksize) / self.stride) + 1
        out_w = int((x.shape[3] - self.ksize) / self.stride) + 1
        output = np.zeros((x.shape[0], self.out_channel, out_h, out_w))

        for bat in range(x.shape[0]):
            for out_ch in range(self.out_channel):
                offset_h = 0
                for h in range(out_h):
                    offset_h = h * self.stride
                    offset_w = 0
                    for w in range(out_w):
                        offset_w = w * self.stride
                        output[bat][out_ch][h][w] = np.sum(x[bat][:, offset_h:offset_h+self.ksize, offset_w:offset_w+self.ksize] * self.kernel[out_ch])
            
            output[bat] += self.bias

        return output

class Convolution2DFast:
    def __init__(self, in_channel, out_channel, ksize=3, stride=2, pad=0):
        self.pad = pad
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.ksize = ksize
        self.kernel = np.random.randn(out_channel, in_channel, ksize, ksize) * 0.01
        self.bias = np.zeros(out_channel)

    def forward(self, x):
        batch_size, in_channel, in_height, in_width = x.shape

        out_h = int((in_height + 2*self.pad - self.ksize) / self.stride) + 1
        out_w = int((in_width + 2*self.pad - self.ksize) / self.stride) + 1

        self.x = x
        self.input_col = im2col(x, self.ksize, self.ksize, self.stride, self.pad) # (batch_size, in_channel, in_h, in_w) -> (batch_size * out_h * out_w, ksize * ksize * in_channel) 
        self.weight_col = self.kernel.reshape(self.out_channel, -1).T # (out_channel, in_channel, ksize, ksize) -> (in_channel * ksize * ksize, out_channel)
        out = np.dot(self.input_col, self.weight_col) + self.bias # (batch_size * out_h * out_w, ksize * ksize * in_channel) * (in_channel * ksize * ksize, out_channel)
        # 結果(batch_size * out_h * outw, out_channel)の2次元配列になってる。biasのshapeは(out_channel)で、1つのout_channelに対して1回加算なので、そのまま加算可能

        out = out.reshape(batch_size, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1) # (batch_size, out_channel, out_h, out_w) -> (batch_size, out_h, out_w, out_channel)
        dout = dout.reshape(-1, self.out_channel) # (batch_size, out_h, out_w, out_channel) -> (batch_size * out_h * out_w, out_channel)

        self.db = np.sum(dout, axis=0) # (batch_size * out_h * outw)分を全部合計して、(out_channel)個のbias勾配を計算
        self.dW = np.dot(self.input_col.T, dout) # (ksize * ksize * in_channel, batch_size * out_h * out_w) x (batch_size * out_h * out_w, out_channel) = (ksize * ksize * in_channel, out_channel)
        self.dW = self.dW.transpose(1, 0).reshape(self.out_channel, self.in_channel, self.ksize, self.ksize) # (ksize * ksize * in_channel, out_channel) -> (out_channel, in_channel, ksize, ksize)

        dcol = np.dot(dout, self.weight_col.T) # (batch_size * out_h * out_w, out_channel) x (out_channel, in_channel * ksize * ksize) = (batch_size * out_h * out_w, in_channel * ksize * ksize)
        dx = col2im(dcol, self.x.shape, self.ksize, self.ksize, self.stride, self.pad)

        return dx
