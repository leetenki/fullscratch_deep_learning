import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    x[x < 0] = 0
    return x

x = np.random.rand(1000, 100) - 0.5 # 1バッチ1000サンプルとして、各サンプルは100個のデータがある
#x = np.random.randn(1000, 100)
layer_size = 5
node_num = 100
activations = {}

for i in range(layer_size):
    if i != 0:
        x = activations[i-1]

    #w = np.random.rand(node_num, node_num) - 0.5
    #w = np.random.randn(node_num, node_num) * 1 # 100in 1000outのlayerを想定
    #w = np.random.randn(node_num, node_num) / np.sqrt(node_num) # Xavierの初期値
    w = np.random.randn(node_num, node_num) * np.sqrt(2 / node_num) # Heの初期値
    z = np.dot(x, w)
    a = relu(z)
    #a = sigmoid(z)
    activations[i] = a

# aを可視化
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1) # 画面を5分割して、第i区分を描画
    plt.title(str(i+1) + "-layer") # タイトル表示
    plt.hist(a.flatten(), 30, range=(0, 1)) # a.flatten()で1次元配列に落とし込んで表示させる。x軸の0〜1の区間を30分割する。

plt.show()
