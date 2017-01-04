import sys, os
sys.path.append(os.pardir)
from scratchnet.links import *
from scratchnet.functions import *
from scratchnet.optimizers import *
import numpy as np
from dataset.mnist import load_mnist
from collections import OrderedDict

class ConvNet:
    def __init__(self):
        self.layers = OrderedDict()

        self.layers["conv"] = Convolution2DFast(1, 30, 5, pad=0, stride=1)
        self.layers["relu1"] = ReLU()
        self.layers["pooling"] = Pooling(2, 2, pad=0, stride=2)
        self.layers["local"] = LocallyConnectedLayer(30, 30, 12, 2, ksize=4, stride=2)
        self.layers["relu2"] = ReLU()
        self.layers["l1"] = Affine(30 * 5 * 5, 100)
        self.layers["relu3"] = ReLU()
        self.layers["l2"] = Affine(100, 10)
        self.layers["softmax_loss"] = SoftmaxCrossEntropy()

    def predict(self, x):
        h = self.layers["conv"].forward(x) # (batch_size, 1, 28, 28) -> (batch_size, 30, 24, 24)
        h = self.layers["relu1"].forward(h)
        h = self.layers["pooling"].forward(h) # (batch_size, 30, 24, 24) -> (batch_size, 30, 12, 12)
        h = self.layers["local"].forward(h) # (batch_size, 30, 12, 12) -> (batch_size, 30, 5, 5)
        h = self.layers["relu2"].forward(h)
        h = self.layers["l1"].forward(h)
        h = self.layers["relu3"].forward(h)
        y = self.layers["l2"].forward(h)

        return y

    def forward(self, x, t):
        y = self.predict(x)
        return self.layers["softmax_loss"].forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def backward(self):
        dout = 1
        dout = self.layers["softmax_loss"].backward()
        dout = self.layers["l2"].backward(dout)
        dout = self.layers["relu3"].backward(dout)
        dout = self.layers["l1"].backward(dout)
        dout = self.layers["relu2"].backward(dout)
        dout = self.layers["local"].backward(dout)
        dout = self.layers["pooling"].backward(dout)
        dout = self.layers["relu1"].backward(dout)
        dout = self.layers["conv"].backward(dout)

        return dout

# load data
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=True, one_hot_label=True)

# ハイパーパラメータ
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

# init model
model = ConvNet()
optimizer = AdaGrad(model, learning_rate)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size) # train_size(60000以下)から、batch_size(100)個を選ぶ
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    loss = model.forward(x_batch, t_batch)
    model.backward()
    optimizer.update()

    print(i+1, model.accuracy(x_batch, t_batch))
