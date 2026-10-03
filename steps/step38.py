import numpy as np

import dezero.functions as F
from dezero import Variable


np.random.seed(1234)


def test_numpy_reshape() -> None:
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.reshape(x, (6,))
    print(y)

    x = np.random.rand(1, 2, 3)
    y1 = x.reshape([2, 3])
    y2 = x.reshape(2, 3)
    print(y1)
    print(y2)


def func(*shape: int | list[int] | tuple[int, ...]):
    if len(shape) == 1:
        if isinstance(shape[0], tuple):
            shape = shape[0]
        elif isinstance(shape[0], list):
            shape = tuple(shape[0])
    print(shape)


def test_variable_reshape() -> None:
    x = Variable(np.random.randn(1, 2, 3))
    y1 = x.reshape(2, 3)
    print(y1)
    y2 = x.reshape((2, 3))
    print(y2)
    y3 = x.reshape([2, 3])
    print(y3)


def test_numpy_transpose() -> None:
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.transpose(x)
    print(y)

    A, B, C, D = 1, 2, 3, 4
    x = np.random.randn(A, B, C, D)
    y = x.transpose(1, 0, 3, 2)
    print(x.shape)
    print(y.shape)


def test_variable_trasnpoese() -> None:
    x = Variable(np.random.randn(2, 3))
    y1 = x.transpose()
    y2 = x.T
    print(y1)
    print(y2)

    A, B, C, D = 1, 2, 3, 4
    x = Variable(np.random.randn(A, B, C, D))
    y1 = x.transpose(1, 0, 3, 2)
    y2 = x.transpose([1, 0, 3, 2])
    y3 = x.transpose((1, 0, 3, 2))
    print(x.shape)
    print(y1.shape)
    print(y2.shape)
    print(y3.shape)


if __name__ == "__main__":
    test_numpy_reshape()

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.reshape(x, (6,))
    y.backward(retain_grad=True)
    print(x.grad)

    func(1, 2)
    func((1, 2))
    func([1, 2])

    test_variable_reshape()

    test_numpy_transpose()

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.transpose(x)
    y.backward()
    print(x.grad)

    test_variable_trasnpoese()
