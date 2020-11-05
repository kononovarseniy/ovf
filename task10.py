from math import *
import matplotlib.pyplot as plt
import numpy as np
from task9 import ThreeDiag


def make_matrix(dt, dx, n):
    def gen():
        t = dt / dx ** 2
        t2 = t / 2
        yield 0, 1 + t, -t2
        for i in range(2, n - 1):
            yield -t2, 1 + t, -t2
        yield -t2, 1 + t, 0

    return ThreeDiag(gen())


def make_d(dt, dx, v):
    def gen():
        t = dt / dx ** 2 / 2
        for i in range(1, len(v) - 1):
            yield v[i] + t * (v[i - 1] - 2 * v[i] + v[i + 1])

    return list(gen())


if __name__ == '__main__':
    a = 0
    b = 1
    n = 100
    def u0(xs):
        return (xs - a) * (1 - xs/b) ** 2        

    dt = 0.03
    xs, dx = np.linspace(a, b, n + 1, retstep=True)
    t = 0
    vs = u0(xs)
    ts = [0]
    ms = [np.max(vs)]
    mms = [np.max(vs)]
    plt.plot(xs, vs, ':r')

    m = make_matrix(dt, dx, n)

    for i in range(5000):
        t += dt
        vs = np.concatenate(([0], m.solve(make_d(dt, dx, vs)), [0]))
        ts.append(t)
        ms.append(np.max(vs))
        mms.append(np.max(np.abs(vs)))
        if i < 100:
            plt.plot(xs, vs)
    plt.show()

    plt.plot(ts, ms)
    plt.plot(ts, mms)
    plt.yscale('log')
    plt.show()


