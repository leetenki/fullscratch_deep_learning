def numerical_diff(f, x):
    h = 10e-8
    return (f(x+h) - f(x-h)) / (2*h)

def func(x):
    return x**2

print(numerical_diff(func, 10))
