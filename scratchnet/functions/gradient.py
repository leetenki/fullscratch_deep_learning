import numpy as np

def numerical_grad(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

# xはn次元配列。f(x)も次元数不定
# xの要素のうち各要素を微小変更させた時の、f(x)の全ての変化量の合計を求める
def partial_numerical_grad(f, x):
    h = 1e-4
    shape = x.shape
    x = x.reshape(-1)
    grads = np.zeros(x.shape)

    for i in range(x.size):
        temp = x[i]
        x[i] += h
        x = x.reshape(shape) # xそのものを書き換える
        dy = f(x) 
        x = x.reshape(-1)
        x[i] -= h
        x = x.reshape(shape) # xそのものを書き換える
        dy -= f(x)
        grads[i] = np.sum(dy / (2*h)) # 誤差関数の結果f(x)は基本的にスカラー値だが、一応sumを取る
        x = x.reshape(-1)
        x[i] = temp

    return grads.reshape(shape)
