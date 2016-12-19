import sys, os
import numpy as np
import pickle
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x -= np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()
batch_size = 100
accuracy_cnt = 0

for i in np.arange(0, len(x), batch_size):
    y = predict(network, x[i:i+batch_size])
    p = np.argmax(y, axis=1)
    accuracy_cnt += np.sum(t[i:i+batch_size] == p)

print(accuracy_cnt / len(x))
