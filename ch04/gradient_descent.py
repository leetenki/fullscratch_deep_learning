import numpy as np

def partial_numerical_diff(f, x):
    h = 1e-4
    return (
        (f(x[0]+h, x[1]) - f(x[0]-h, x[1])) / (2*h),
        (f(x[0], x[1]+h) - f(x[0], x[1]-h)) / (2*h)
    )

def func(x0, x1):
    return x0 ** 2 + x1 ** 2

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        print(x)
        grad = partial_numerical_diff(f, x)
        x -= lr * grad
    import pdb
    pdb.set_trace()
    return x

print(gradient_descent(func, (100.0, 100.0)))
