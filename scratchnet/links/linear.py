import numpy as np

class Linear:
    def __init__(self, n_in, n_out, weight_decay_rate=0.1):
        self.W = np.random.randn(n_out, n_in) * np.sqrt(2 / n_in) # xavierの初期値
        self.b = np.zeros(n_out)
        self.weight_decay_rate = weight_decay_rate
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W.T) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W)
        self.dW = np.dot(dout.T, self.x) + self.weight_decay_rate * self.W # with weight_decay
        self.db = np.sum(dout, axis=0) + self.weight_decay_rate * self.b # with weight_decay

        return dx

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
