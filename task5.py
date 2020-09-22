import matplotlib.pyplot as plt
import numpy as np
import math
from functools import partial
from time import time

def get_newton_value(ps, xs, x):
    res = ps[-1]
    for xi, p in zip(xs[-2::-1], ps[-2::-1]):
        res *= x - xi
        res += p
    return res

def get_newton_ps(xs, ys):
    ps = [ys[0]]
    t1 = ys
    for i in range(1, len(xs)):
        t2 = []
        for j in range(len(t1) - 1):
            t2.append((t1[j] - t1[j + 1]) / (xs[j] - xs[j + i]))
        t1 = t2
        ps.append(t1[0])
    
    return ps

def newton_poly(xs, ys):
    xs = list(xs)
    ys = list(ys)
    ps = get_newton_ps(xs, ys)
    return partial(get_newton_value, ps, xs)

if __name__ == '__main__':
    func = np.vectorize(math.log)
    n = int(input('n: '))
    xs = np.linspace(1.0, 2.0, n + 1)
    ys = func(xs)


    print('Starting')
    start = time()

    poly = np.vectorize(newton_poly(xs, ys))

    xs_dense = np.linspace(0.5, 2.5, 2000)
    ys_func = func(xs_dense)
    ys_poly = poly(xs_dense)
    es = np.abs(ys_func - ys_poly)

    print('Done.\nElapsed time: ', time() - start)

    plt.subplot(2, 1, 1)
    plt.title('Interpolation result')
    plt.plot(xs, ys, 'or')
    plt.plot(xs_dense, ys_func, '--')
    plt.plot(xs_dense, ys_poly)
    plt.ylabel('f(x)')

    plt.subplot(2, 1, 2)
    plt.plot(xs_dense, es)
    plt.ylabel('Absolute error')
    plt.yscale('log')
    plt.xlabel('x')
    plt.show()
