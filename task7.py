from math import *
import matplotlib.pyplot as plt
import numpy as np

from task6 import rk2

if __name__ == '__main__':
    a = 10
    b = 2
    c = 2
    d = 10
    def func(t, x):
        x0 = a * x[0] - b * x[0] * x[1]
        x1 = c * x[0] * x[1] - d * x[1]
        return np.array([x0, x1])


    for i in range(1, 5):
        ts, xs = rk2(func, 0, 3, np.array([i, i]), 500)
        plt.plot([x[0] for x in xs], [x[1] for x in xs])
    ts, xs = rk2(func, 0, 3, np.array([4.9, 4.9]), 500)
    plt.plot([x[0] for x in xs], [x[1] for x in xs])
    plt.show()


