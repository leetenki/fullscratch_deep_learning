import numpy as np

class Linear:
    def __init__(self, n_in, n_out):
        self.W = 0.01 * np.random.randn(n_out, n_in)
        self.b = np.zeros(n_out)
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W.T) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W)
        self.dW = np.dot(dout.T, self.x)
        self.db = np.sum(dout, axis=0)

        return dx

    def update(self, epsilon=0.1):
        self.W = self.W - self.dW * epsilon
        self.b = self.b - self.db * epsilon

class Affine:
    def __init__(self, n_input, n_output, weight_init_std=0.01): 
        self.W = weight_init_std * np.random.randn(n_input, n_output)
        self.b = np.zeros(n_output)
    
        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x): 
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1) 
        self.x = x 

        out = np.dot(self.x, self.W) + self.b

        return out 

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
          
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx

    def update(self, epsilon=0.1):
        self.W -= self.dW * epsilon
        self.b -= self.db * epsilon
