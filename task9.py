from math import *
import matplotlib.pyplot as plt
import numpy as np

@np.vectorize
def f(x):
    return sin(x)

class ThreeDiag:
    def __init__(self, lines):
        self.a = []
        self.b = []
        self.c = []
        for ta, tb, tc in lines:
            self.a.append(ta)
            self.b.append(tb)
            self.c.append(tc)

    def solve(self, d):
        a = np.copy(self.a)
        b = np.copy(self.b)
        c = np.copy(self.c)
        d = np.copy(d)

        for i in range(1, len(a)):
            k = a[i] / b[i-1]
            b[i] -= k*c[i-1]
            d[i] -= k*d[i-1]

        d[-1] /= b[-1]
        for i in reversed(range(0, len(a) - 1)):
            d[i] = (d[i] - c[i]*d[i+1])/b[i]

        return d


def make_matrix(n, h):
    """
    n is the number of segments
    h is the length of each segment
    """
    r = (1/h**2, -2/h**2, 1/h**2)

    def gen():
        yield 0, 1, 0
        for i in range(n - 1):
            yield r
        yield 0, 1, 0

    return ThreeDiag(gen())

def make_d(xs, f, ya, yb):
    return np.concatenate(([ya], f(xs[1:-1]), [yb]))

if __name__ == '__main__':
    n = 1000
    a = -pi
    b = pi
    xs, h = np.linspace(a, b, n, retstep=True)
    m = make_matrix(n-1, h)

    d1 = make_d(xs, f, 0, 0)
    d2 = make_d(xs, f, -10, 10)
    d3 = make_d(xs, f, 10, -10)
    d4 = make_d(xs, f, 11, -9)
    d5 = make_d(xs, f, 1, 1)

    y1 = m.solve(d1)
    y2 = m.solve(d2)
    y3 = m.solve(d3)
    y4 = m.solve(d4)
    y5 = m.solve(d5)

    plt.plot(xs, y1)
    plt.plot(xs, y2)
    plt.plot(xs, y3)
    plt.plot(xs, y4)
    plt.plot(xs, y5)
    plt.show()

