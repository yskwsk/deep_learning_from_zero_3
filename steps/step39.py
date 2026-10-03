import numpy as np

import dezero.functions as F
from dezero import Variable


def test_sum1() -> None:
    x = Variable(np.array([1, 2, 3, 4, 5, 6]))
    y = F.sum(x)
    y.backward()
    print(y)
    print(x.grad)

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.sum(x)
    y.backward()
    print(y)
    print(x.grad)


def test_numpy_sum() -> None:
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.sum(x, keepdims=True)
    print(y)
    print(y.shape)


def test_sum2() -> None:
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.sum(x, axis=0)
    y.backward()
    print(y)
    print(x.grad)

    np.random.seed(1234)
    x = Variable(np.random.randn(2, 3, 4, 5))
    y = x.sum(keepdims=True)
    print(y.shape)


if __name__ == "__main__":
    test_sum1()

    test_numpy_sum()

    test_sum2()
