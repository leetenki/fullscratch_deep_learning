import sys, os
sys.path.append(os.pardir)
from scratchnet.functions import *
import numpy as np
from dataset.mnist import load_mnist

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(hidden_size, input_size)
        self.params["b1"] = weight_init_std * np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(output_size, hidden_size)
        self.params["b2"] = weight_init_std * np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1.T) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2.T) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}

        grads["W1"] = partial_numerical_grad(loss_W, self.params["W1"]) # W1を微小変化させる
        grads["b1"] = partial_numerical_grad(loss_W, self.params["b1"]) # b1を微小変化させる
        grads["W2"] = partial_numerical_grad(loss_W, self.params["W2"]) # W2を微小変化させる
        grads["b2"] = partial_numerical_grad(loss_W, self.params["b2"]) # b2を微小変化させる

        return grads


# load data
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

train_loss_list = []

# ハイパーパラメータ
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.2

# init model
net = TwoLayerNet(784, 32, 10)

for i in range(iters_num):
    # ミニバッチ作成
    batch_mask = np.random.choice(train_size, batch_size) # train_size(60000以下)から、batch_size(100)個を選ぶ
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配計算して重み更新
    grad = net.numerical_gradient(x_batch, t_batch)
    for key in ("W1", "b1", "W2", "b2"):
        net.params[key] -= learning_rate * grad[key]

    loss = net.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print(loss)
