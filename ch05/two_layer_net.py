import sys, os
sys.path.append(os.pardir)
from scratchnet.links import *
from scratchnet.functions import *
import numpy as np
from dataset.mnist import load_mnist
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}

        self.layers = OrderedDict()
        self.layers["l1"] = Linear(input_size, hidden_size)
        self.layers["relu1"] = ReLU()
        self.layers["l2"] = Linear(hidden_size, output_size)
        self.layers["softmax_loss"] = SoftmaxCrossEntropy()

    def predict(self, x):
        h = self.layers["l1"].forward(x)
        h = self.layers["relu1"].forward(h)
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
        dout = self.layers["relu1"].backward(dout)
        dout = self.layers["l1"].backward(dout)

        return dout

    def numerical_grad(self, x, t):
        loss_W = lambda W: self.forward(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.layers["l1"].W)
        grads['b1'] = numerical_gradient(loss_W, self.layers["l1"].b)
        grads['W2'] = numerical_gradient(loss_W, self.layers["l2"].W)
        grads['b2'] = numerical_gradient(loss_W, self.layers["l2"].b)

        return grads

    def update(self, epsilon):
        self.layers["l1"].update(epsilon)
        self.layers["l2"].update(epsilon)

# load data
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

# ハイパーパラメータ
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

# init model
model = TwoLayerNet(784, 1000, 10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size) # train_size(60000以下)から、batch_size(100)個を選ぶ
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    model.forward(x_batch, t_batch)
    model.backward()

    # numerical_gradient check
    #numerical_grads = model.numerical_grad(x_batch, t_batch)
    #print(np.sum(numerical_grads["W1"] - model.layers["l1"].dW))
    #print(np.sum(numerical_grads["W2"] - model.layers["l2"].dW))
    #print(np.sum(numerical_grads["b1"] - model.layers["l1"].db))
    #print(np.sum(numerical_grads["b2"] - model.layers["l2"].db))

    model.update(learning_rate)

    print(model.accuracy(x_batch, t_batch))
