from math import *
from itertools import islice
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

def generic_method(get_next, f, t0, t1, x0, n):
    ts, h = np.linspace(t0, t1, n, retstep=True)
    x = x0
    xs = [x]
    for i, t in enumerate(ts[:-1]):
        x = get_next(f, t, x, h)
        xs.append(x)
    return ts, np.array(xs)

def euler_get_next(f, t, x, h):
    return x + h * f(t, x)

def rk2_get_next(f, t, x, h, *, a):
    a1 = h * (1 - a)
    a2 = h * a
    b1 = h / (2 * a)
    b2 = h / (2 * a)
    return x + a1 * f(t, x) + a2 * f(t + b1, x + b2 * f(t, x))

def rk4_get_next(f, t, x, h):
    k1 = f(t, x)
    k2 = f(t + h/2, x + h/2 * k1)
    k3 = f(t + h/2, x + h/2 * k2)
    k4 = f(t + h, x + h * k3)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def abs_diff(a, b):
    return np.abs(a - b)

euler = partial(generic_method, euler_get_next)
rk2 = partial(generic_method, partial(rk2_get_next, a=3/4.))
rk2a = partial(generic_method, partial(rk2_get_next, a=1/2.))
rk2b = partial(generic_method, partial(rk2_get_next, a=1.))
rk4 = partial(generic_method, rk4_get_next)

if __name__ == '__main__':
    #
    # dx
    # -- = -x,  x(0) = 1,  0 < t < 3
    # dt
    #

    @np.vectorize
    def solution(t):
        return exp(-t)

    def func(t, x):
        return -x
    
    methods = [
        (euler, 'Euler'),
        (rk2, 'Runge-Kutta 2-nd order'),
        (rk2a, 'Runge-Kutta 2-nd order a = 1/2'),
        (rk2b, 'Runge-Kutta 2-nd order a = 1'),
        (rk4, 'Runge-Kutta 4-th order')
    ]

    results = [m(func, 0, 3, 1, 100) for m, _ in methods]
    ts = results[0][0]
    ss = solution(ts)
    #ts, ss = rk4(func, 0, 3, 1, 100)
    results = [xs for _, xs in results]
    errors = [abs_diff(xs, ss) for xs in results]

    plt.subplot(2, 1, 1)
    for xs, (_, name) in zip(results, methods):
        plt.plot(ts, xs, label=name)
    plt.plot(ts, ss, ':', label='Analitical')
    plt.legend()

    plt.subplot(2, 1, 2)
    for xs, (_, name) in zip(errors, methods):
        plt.plot(ts, xs, label=name)
    plt.yscale('log')
    plt.legend()
    plt.show()
