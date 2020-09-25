import numpy as np
from scipy.special import jv
from functools import partial
from math import *

from task3 import int_simpson

def magic_func(m, x, t):
    return cos(m*t-x*sin(t))/pi

def my_jv(m, x):
    # Функция у нас хорошая, четвертая производная не очень велика
    # Поэтому при скорости сходимости 1/N^4
    # значения N=100 вполне достаточно для точности 10^-15
    return int_simpson(partial(magic_func, m, x), 0, pi, 100)

def num_diff(func, dx):
    # Метод даёт погрешность < f'''dx^2
    return lambda x: (func(x + dx) - func(x - dx)) / (2 * dx)

J0 = np.vectorize(partial(my_jv, 0))
J1 = np.vectorize(partial(my_jv, 1))
J0D = np.vectorize(num_diff(partial(my_jv, 0), 1e-5))

if __name__ == '__main__':
    xs, h = np.linspace(0, 2*pi, 100, retstep=True)

    j0s = J0(xs)
    print(np.max(np.abs(j0s - jv(0, xs))))

    j1s = J1(xs)
    print(np.max(np.abs(j1s - jv(1, xs))))

    j0ds = J0D(xs)
    print(np.max(np.abs(j1s + j0ds)))

