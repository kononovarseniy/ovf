import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy as np
from math import *

def func1(x):
    return 1/(1 + x*x)

def func2(x):
    return x**(1/3.0)*exp(sin(x))

def int_trapezoidal(f, a, b, n):
    h = (b - a) / n
    s = (f(a) + f(b)) / 2
    for i in range(1, n):
        s += f(a + h * i)
    return s * h

def int_simpson(f, a, b, n):
    if n % 2 == 1:
        n += 1

    h = (b - a) / n
    s = f(a) + f(b)
    n2 = int(n/2)
    for i in range(1, n2):
        s += 2 * f(a + h * 2 * i)
    for i in range(1, n2 + 1):
        s += 4 * f(a + h * (2 * i - 1))
    return s * h / 3

def main(func, r_a, r_b, N):
    xs = np.linspace(r_a, r_b, 1000)
    ys = np.vectorize(func)(xs)
    print(sum(ys[-2::-1]) * (r_b - r_a) / 1000)
    print(sum(ys[1::]) * (r_b - r_a) / 1000)
    plt.plot(xs, ys)
    plt.show()

    integral = quad(func, r_a, r_b)
    print(integral)
    integral = integral[0]


    ns = []
    t_points = []
    s_points = []

    print('/------+-----------+-----------+-----------+-----------\\')
    print('|   N  | trapezoid |   error   |  simpson  |   error   |')
    print('+------+-----------+-----------+-----------+-----------+')
    for i in range(N):
        n = 2 ** i
        ns.append(n)
        int_t = int_trapezoidal(func, r_a, r_b, n)
        err_t = abs(int_t - integral)
        t_points.append(err_t)
        int_s = int_simpson(func, r_a, r_b, n)
        err_s = abs(int_s - integral)
        s_points.append(err_s)
        print(f'| {n:>4} | {int_t:>9.6g} | {err_t:>9.3e} | {int_s:>9.6g} | {err_s:>9.3e} |')
    print('\\------+-----------+-----------+-----------+-----------/')

    plt.plot(ns, t_points, label='trapezoidal')
    plt.plot(ns, s_points, label='simpson')
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

#print(quad(func2, 0, 0.001))
#print(int_simpson(func2, 0, 0.001, 2))
#main(func1, -1, 1, 13)
main(func2, 0.01, 1, 20)


