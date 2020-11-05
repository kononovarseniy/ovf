from math import *
import cmath
import matplotlib.pyplot as plt
import numpy as np


def dft(xs, reverse=False):
    n = len(xs)
    ys = [None] * n
    tmp = 2j * pi / n * (-1 if reverse else 1)
    tmp2 = sqrt(n)
    for j in range(n):
        s = 0
        for k in range(n):
            s += xs[k] * cmath.exp(tmp * j * k)
        ys[j] = s / tmp2
    return ys

def h_window(ks):
    return 1/2 * (1 - np.cos(2*pi*ks/n))

def handle_transformed(ys):
    n = len(ys)
    return np.abs(np.concatenate([ys[(n+1)//2:], ys[0:n//2]]))

if __name__ == '__main__':
    a0 = 1
    w0 = 5.1
    a1 = 0.002
    w1 = 25.5
    n = 100
    T = 2 * pi
    t_step = T / n
    W = 2 * pi / t_step
    w_step = W / n

    def func(ts):
        return a0 * np.sin(w0*ts) + a1*np.sin(w1*ts)

    ks = np.array(list(range(n)))
    ts = t_step * ks
    ws = w_step * (ks - n // 2)
    wnd = h_window(ks)

    xs = func(ts)

    ys1 = dft(xs)
    xs1 = dft(ys1, reverse=True)

    ys2 = dft(xs*wnd)
    xs2 = dft(ys2, reverse=True)
    
    fig, axs = plt.subplots(2, 3, sharey='col')
    for r in range(2):
        for c in range(3):
            axs[r, c].grid(b=True, which='both', axis='both')

    axs[0, 0].plot(ts, xs)
    axs[0, 2].plot(ts, xs1)
    axs[0, 1].plot(ws, handle_transformed(ys1))
    axs[0, 1].set_yscale('log')

    axs[1, 0].plot(ts, xs)
    axs[1, 2].plot(ts, xs2)
    axs[1, 1].plot(ws, handle_transformed(ys2))
    axs[1, 1].set_yscale('log')

    plt.show()

