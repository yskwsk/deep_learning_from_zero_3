import numpy as np
from dezero import SimpleVariable as Variable


def rosenbrock(x0: Variable, x1: Variable) -> Variable:
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y


if __name__ == "__main__":
    # x0 = Variable(np.array(0.0))
    # x1 = Variable(np.array(2.0))

    # y = rosenbrock(x0, x1)
    # y.backward()
    # print(x0.grad, x1.grad)

    x0 = Variable(np.array(0.0))
    x1 = Variable(np.array(2.0))
    lr = 0.001
    iters = 1000

    for i in range(iters):
        print(x0, x1)

        y = rosenbrock(x0, x1)

        x0.cleargrad()
        x1.cleargrad()
        y.backward()

        x0.data -= lr * x0.grad if x0.grad is not None else 0.0
        x1.data -= lr * x1.grad if x1.grad is not None else 0.0
