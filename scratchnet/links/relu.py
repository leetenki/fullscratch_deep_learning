import numpy as np

class ReLU:
    def __init__(self):
        self.mask = None

    # xはnumpy配列
    def forward(self, x):
        self.mask = (x <= 0) # forward時にmaskを記憶しておく
        out = x.copy()
        out[self.mask] = 0

        return out

    # doutはnumpy配列
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
