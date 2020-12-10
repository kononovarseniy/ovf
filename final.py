from math import exp
from time import time

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

from crank_nicolson_fast import Solver


def main():
    L = 1

    # def v0(x, y):
    #     return (1 - x ** 2 / L ** 2) * (1 - y ** 2 / L ** 2)

    def v0(x, y):
        r2 = x ** 2 + y ** 2
        if r2 > 0.25:
            return 0
        return 2 * exp(-1 / (1 - 4 * r2))

    dt = 0.01

    nx = 200
    ny = 200
    x0 = -L
    x1 = L
    y0 = -L
    y1 = L

    x_solvers = [Solver(x0, x1, nx, dt / 2, lambda x: 3) for _ in range(ny)]
    y_solvers = [Solver(y0, y1, ny, dt / 2, lambda x: 0.5) for _ in range(nx)]

    xx, yy = np.meshgrid(x_solvers[0].xs, y_solvers[0].xs)
    zz = []
    vv = np.vectorize(v0, otypes=[float])(xx, yy)

    start = time()
    for i in range(40):
        zz.append(np.copy(vv))

        for j, s in enumerate(x_solvers):
            vv[j + 1, :] = s.do_step(vv[j + 1, :])
        for j, s in enumerate(y_solvers):
            vv[:, j + 1] = s.do_step(vv[:, j + 1])

    print(time() - start)

    def update_plot(frame):
        nonlocal plot
        plot.remove()
        plot = ax.plot_surface(xx, yy, zz[frame], cmap='magma', vmin=0, vmax=1)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_zlim(0, 1)
    plot = ax.plot_surface(xx, yy, vv)
    anim = animation.FuncAnimation(fig, update_plot, len(zz), interval=1000 / 30)
    plt.show()


if __name__ == '__main__':
    main()
