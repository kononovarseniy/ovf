import numpy as np

from task9 import ThreeDiag


def prev_cur_next(xs):
    xs = iter(xs)
    px = next(xs)
    cx = next(xs)
    for nx in xs:
        yield px, cx, nx
        px = cx
        cx = nx


def _gen_matrix(xs, ks, dt, dx):
    k_ac = -dt / (2 * dx ** 2)
    k_b = dt / dx ** 2

    yield 0, 1, 0
    for k in ks[1:-1]:
        yield k_ac * k, 1 + k_b * k, k_ac * k
    yield 0, 1, 0


def _gen_d(vs, ks, dt, dx):
    k_d = dt / dx ** 2 / 2

    yield 0
    for k, (pv, cv, nv) in zip(ks, prev_cur_next(vs)):
        yield cv + k_d * k * (pv - 2 * cv + nv)
    yield 0


class Solver:
    def __init__(self, x0, x1, num, dt, func):
        self.dx = (x1 - x0) / (num + 1)
        self.dt = dt
        self.num = num
        self.xs = np.linspace(x0, x1, num + 2)
        self.ks = np.vectorize(func)(self.xs)

        self.matrix = ThreeDiag(_gen_matrix(self.xs, self.ks, self.dt, self.dx))

    def do_step(self, vs):
        assert len(vs) == self.num + 2
        ds = np.fromiter(_gen_d(vs, self.ks, self.dt, self.dx), float, count=self.num + 2)
        return self.matrix.solve(ds)
