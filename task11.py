from math import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from task9 import ThreeDiag
from abc import ABC, abstractmethod


class Operator(ABC):
    @abstractmethod
    def gen_lines(self, xs):
        pass

    def get_matrix(self, xs):
        return ThreeDiag(self.gen_lines(xs))

    def __add__(self, op):
        return OperatorSum(self, op)

    def __mul__(self, const):
        return OperatorMul(self, const)

class OperatorSum(Operator):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def gen_lines(self, xs):
        for (a1, b1, c1), (a2, b2, c2) in zip(self.a.gen_lines(xs), self.b.gen_lines(xs)):
            yield a1 + a2, b1 + b2, c1 + c2

class OperatorMul(Operator):
    def __init__(self, op, const):
        self.op = op
        self.const = const

    def gen_lines(self, xs):
        k = self.const
        return ((a * k, b * k, c * k) for a, b, c in self.op.gen_lines(xs))

class FunctionOperator(Operator):
    def __init__(self, func):
        self.func = func

    def gen_lines(self, xs):
        return ((0, self.func(x), 0) for x in xs)

class IdentityOperator(FunctionOperator):
    def __init__(self):
        super.__init__(lambda x: 1)

class SecondDerivativeOperator(Operator):
    def gen_lines(self, xs):
        h = xs[1] - xs[0]

        yield 0, 1, 0
        for x in xs[1:-1]:
            yield 1/h**2, -2/h**2, 1/h**2
        yield 0, 1, 0


def polynomial_guess(xs, order):
    a = xs[0]
    b = xs[-1]
    zeros = np.linspace(a, b, order + 2)
    def gen():
        for x in xs:
            val = -1
            for z in zeros:
                val *= (x - z)
            yield val

    u = np.array(list(gen()))
    norm = la.norm(u)
    return u / norm

if __name__ == '__main__':
    op = SecondDerivativeOperator() * (-1/2) + FunctionOperator(lambda x: 1/2 * x**2)

    a = -100
    b = 100
    n = 10000
    delta = 1e-5

    xs = np.linspace(a, b, n)
    m = op.get_matrix(xs)

    prev_u = polynomial_guess(xs, 0)
    prev_norm = 1
    plt.plot(xs, prev_u, '--k', label='0')

    curr_u = m.solve(prev_u)
    curr_norm = la.norm(curr_u)
    plt.plot(xs, curr_u / curr_norm, label=1)
    i = 2

    curr_val = prev_norm / curr_norm
    while True:
        prev_u = curr_u
        prev_norm = curr_norm
        curr_u = m.solve(curr_u)
        curr_norm = la.norm(curr_u)

        prev_val = curr_val
        curr_val = prev_norm / curr_norm

        plt.plot(xs, curr_u / curr_norm, label=str(i))
        i += 1
        if abs(curr_val - prev_val) < delta:
            break

    print(curr_val)
    plt.grid(b=True, which='both', axis='both')
    plt.legend()
    plt.show()
    
        
