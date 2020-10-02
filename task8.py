from math import *
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

from task6 import generic_method

if __name__ == '__main__':
    E = np.identity(2)
    def implicit_euler_get_next(f, t, x, h):
        J = f(t, x, ret='matrix')
        return x + inv(E + h*J) @ (h * f(t, x))
    
    solve = partial(generic_method, implicit_euler_get_next)

    def func(t, x, ret='value'):
        J = np.array([[+998, +1998], [-999, -1999]])
        if ret == 'matrix':
            return J
        elif ret == 'value':
            return J @ x


    for i in range(-5, 10):
        ts, xs = solve(func, 0, 1, np.array([i, i]), 10000)
        plt.plot([x[0] for x in xs], [x[1] for x in xs])
    plt.show()


