import numpy as np

from scipy.linalg import solve_banded


class Solver:
    def __init__(self, x0, x1, num, dt, func):
        self.dx = (x1 - x0) / (num + 1)
        self.dt = dt
        self.num = num
        self.xs = np.linspace(x0, x1, num + 2)
        self.ks = np.vectorize(func)(self.xs)

        k_a = - self.dt / (2 * self.dx ** 2)
        k_b = (1 + self.dt / self.dx ** 2)
        k_c = k_a

        bs = 1 + self.dt / self.dx ** 2 * self.ks[1:-1]

        cc = np.concatenate([[0], [0], k_a * self.ks[1:-1]])
        bb = np.concatenate([[1], bs, [1]])
        aa = np.concatenate([k_c * self.ks[1:-1], [0], [0]])

        self.matrix = np.row_stack([cc, bb, aa])

    def do_step(self, vs):
        assert len(vs) == self.num + 2
        k_d = self.dt / self.dx ** 2 / 2
        ds = np.concatenate([
            [0],
            vs[1:-1] + k_d * self.ks[1:-1] * (vs[:-2] - 2 * vs[1:-1] + vs[2:]),
            [0]
        ])
        return solve_banded((1, 1), self.matrix, ds)
