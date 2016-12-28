import sys, os
sys.path.append(os.pardir)
from scratchnet.links import *
from scratchnet.functions import *
from scratchnet.optimizers import *
import numpy as np
from dataset.mnist import load_mnist
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}

        self.layers = OrderedDict()
        self.layers["l1"] = Linear(input_size, hidden_size)
        self.layers["relu1"] = ReLU()
        self.layers["dropout"] = Dropout()
        self.layers["l2"] = Linear(hidden_size, output_size)
        self.layers["softmax_loss"] = SoftmaxCrossEntropy()

    def predict(self, x, train=True):
        h = self.layers["l1"].forward(x)
        h = self.layers["relu1"].forward(h)
        h = self.layers["dropout"].forward(h, train)
        y = self.layers["l2"].forward(h)

        return y

    def forward(self, x, t):
        y = self.predict(x)
        return self.layers["softmax_loss"].forward(y, t)

    def accuracy(self, x, t, train=True):
        y = self.predict(x, train)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def backward(self):
        dout = 1
        dout = self.layers["softmax_loss"].backward()
        dout = self.layers["l2"].backward(dout)
        dout = self.layers["relu1"].backward(dout)
        dout = self.layers["dropout"].backward(dout)
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

# load data
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

# ハイパーパラメータ
iters_num = 10000
train_size = 300
test_size = x_test.shape[0]
batch_size = 100
learning_rate = 0.1

# init model
model = TwoLayerNet(784, 1000, 10)
optimizer = SGD(model, learning_rate)
#optimizer = Momentum(model, learning_rate)
#optimizer = AdaGrad(model, learning_rate)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size) # train_size(60000以下)から、batch_size(100)個を選ぶ
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    model.forward(x_batch, t_batch)
    train_accuracy = model.accuracy(x_batch, t_batch, train=False)

    # learn
    model.backward()
    optimizer.update()

    # test
    batch_mask = np.random.choice(test_size, batch_size) # train_size(60000以下)から、batch_size(100)個を選ぶ
    x_batch = x_test[batch_mask]
    t_batch = t_test[batch_mask]
    test_accuracy = model.accuracy(x_batch, t_batch, train=False)

    print(train_accuracy, test_accuracy)
