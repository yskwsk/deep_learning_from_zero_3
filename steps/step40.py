import numpy as np

from dezero import Variable
from dezero.utils import sum_to


def test_numpy_broadcast_to() -> None:
    x = np.array([1, 2, 3])
    y = np.broadcast_to(x, (2, 3))
    print(y)

    x0 = np.array([1, 2, 3])
    x1 = np.array([10])
    y = x0 + x1
    print(y)


def test_sum_to() -> None:
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = sum_to(x, (1, 3))
    print(y)

    y = sum_to(x, (2, 1))
    print(y)


if __name__ == "__main__":
    test_numpy_broadcast_to()

    test_sum_to()

    x0 = Variable(np.array([1, 2, 3]))
    x1 = Variable(np.array([10]))
    y = x0 + x1
    print(y)

    y.backward()
    print(x0.grad)
    print(x1.grad)
