import numpy as np
from dezero import Variable


def f(x: Variable) -> Variable:
    y = x ** 4 - 2 * x ** 2
    return y


def test() -> None:
    x = Variable(np.array(2.0))
    y = f(x)
    y.backward(create_graph=True)
    print(y)
    print(x.grad)

    if x.grad is None:
        raise ValueError("x.grad is None")
    gx = x.grad
    x.cleargrad()
    gx.backward()
    print(x.grad)


if __name__ == "__main__":
    # test()

    x = Variable(np.array(2.0))
    iters = 10

    for i in range(iters):
        print(i, x)

        y = f(x)
        x.cleargrad()
        y.backward(create_graph=True)

        if x.grad is None:
            raise ValueError("x.grad is None")

        gx = x.grad
        x.cleargrad()
        gx.backward()
        gx2 = x.grad

        if (x.data is None) or (gx.data is None):
            raise ValueError("x.data or gx.data is None")

        x.data -= gx.data / gx2.data
