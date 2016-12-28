import numpy as np

def relu(x):
    z = x.copy()
    z[x < 0] = 0
    return z

def batch_normalize(x):
    average = np.sum(x, axis=0) / x.shape[0] # 平均の計算
    dispersion = np.sum((x - average) ** 2, axis=0) / x.shape[0] # 分散の計算
    x = (x - average) / np.sqrt(dispersion + 1e-7) # xの更新

    return x

# 入力に対する線形結合変換
h = np.array([[-10, -10, -9, -10], [-1, -4, -2, -3], [-0.1, -0.1, -0.1, -0.1]])

z = relu(h)
print(h)
print("\nrelu ↓\n")
print(z)

print("\n--------------------------------------\n")

print(h)
print("\nbatch_norm ↓\n")
h = batch_normalize(h)
print(h)
print("\nrelu ↓\n")
z = relu(h)
print(z)
