import math
from typing import cast

import numpy as np

from dezero import Function, Variable
from dezero.utils import plot_dot_graph


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


def sin(x) -> Variable:
    return Sin()(x)


def my_sin(x: Variable, threshold: float = 0.0001) -> Variable:
    y = 0
    # y = Variable(np.array(0))
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return cast(Variable, y)


if __name__ == "__main__":
    x = Variable(np.array(np.pi/4.0))
    y = sin(x)
    y.backward()

    print(y.data)
    print(x.grad)

    x1 = Variable(np.array(np.pi/4.0))
    y1 = my_sin(x1, threshold=1e-150)
    y1.backward()

    print(y1.data)
    print(x1.grad)

    x1.name = "x"
    y1.name = "y"
    plot_dot_graph(y1, verbose=False, to_file="my_sin.png")
