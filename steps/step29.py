import numpy as np
from dezero import SimpleVariable as Variable


def f(x: Variable) -> Variable:
    y = x ** 4 - 2 * x ** 2
    return y


def gx2(x: np.ndarray) -> np.ndarray:
    dy2dx2 = 12 * x ** 2 - 4
    return dy2dx2


if __name__ == "__main__":
    x = Variable(np.array(2.0))
    iters = 10

    for i in range(iters):
        print(i, x)

        y = f(x)
        x.cleargrad()
        y.backward()

        if x.grad is not None:
            x.data -= x.grad / gx2(x.data)
