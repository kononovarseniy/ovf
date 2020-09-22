import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root
from math import *

"""
-1/2*F"(x) + (U(x) - E)*F(x) = 0


         ___________         ________
        /  2       '        / 1     '
ctg _  / 2a (1 - x)  -- _  / --- - 1
     \/                  \/   x


x = -E / U_0
"""

def dichotomy(f, a, b, steps):
    if isinstance(a, tuple):
        a, fa = a
        a = float(a)
    else:
        a = float(a)
        fa = f(fa)

    if isinstance(b, tuple):
        b, ba = b
        b = float(b)
    else:
        b = float(b)
        fb = f(b)

    for _ in range(steps):
        c = (a + b) / 2
        fc = f(c)
        if fc * fa < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    
    return a, b

def simple_iter(f, a, b, x0, delta, k = 1, max_steps = 100):
    x2 = None
    x1 = None
    i = 0
    points = [x0]
    while i < max_steps:
        i += 1
        x2 = x1
        x1 = x0
        x0 += -k * f(x0)
        points.append(x0)
        if der(x0) * k > 1:
            print('Bad point')
        if not a <= x0 <= b:
            return None, i, points
        if i >= 2:
            if (x0 - x1)**2 / abs(2 * x1 - x0 - x2) < delta:
                return x0, i, points
    return x0, i, points

def newton(f, d, a, b, x0, delta, max_steps):
    x1 = None
    i = 0
    points = [x0]
    while i < max_steps:
        i += 1
        x1 = x0
        x0 += -f(x0) / d(x0)
        points.append(x0)
        if not a <= x0 <= b:
            return None, i, points
        if abs(x0 - x1) < delta:
            return x0, i, points
    return x0, i, points

def print_result(msg, res, steps):
    print(msg)
    print(f'E = {-res * U0 if res else "???"}, steps: {steps}')

a = float(input('a: '))
U0 = float(input('U0: '))
prec = int(input('decimal places: '))

def func(x):
    return 1 / tan(sqrt(2 * a * a * (1 - x))) - sqrt(1 / x - 1)
def der(x):
    t = sqrt(2*a*a*(1 - x))
    return 1/(2*x*x*sqrt(1/x - 1)) + a*a/sin(t)**2/t
    
n = floor(sqrt(2) / pi * a)
range_a = 0
range_b = 1 - (pi * n / a) ** 2 / 2
x0 = (range_a + range_b) / 2

target = root(func, x0).x

print('left: ', range_a)
print('right:', range_b)

delta = 10 ** -(prec + 1)  / U0
steps = ceil(log2((range_b - range_a) / delta))

res_min, res_max = dichotomy(func, (range_a, -inf), (range_b, inf), steps)
print_result('dichotomy:', (res_min + res_max) / 2, steps)
print('min', -res_max * U0)
print('max', -res_min * U0)

x, i_it, ps_it = simple_iter(func, range_a, range_b, x0, delta, 1/der(x0), max_steps=1000)
print_result('simple iter:', x, i_it)

x, i_ne, ps_ne = newton(func, der, range_a, range_b, x0, delta, max_steps=100)
print_result('newton:', x, i_ne)

plt.plot(range(i_it + 1), np.abs(np.array(ps_it) - target))
plt.plot(range(i_ne + 1), np.abs(np.array(ps_ne) - target))
plt.yscale("log")
plt.show()

xs = np.arange(range_a + 1e-5, range_b, 1/1000)
ys = np.vectorize(func)(xs)
ys2 = np.vectorize(der)(xs)
plt.plot(xs, ys)
plt.plot(xs, ys2)
plt.ylim(-10, 10)
plt.show()
