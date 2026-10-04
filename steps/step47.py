import numpy as np

from dezero import ArrayType, Variable, as_variable
import dezero.functions as F
from dezero.models import MLP


def test_numpy_get_item() -> None:
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y1 = x.__getitem__(1)
    print(y1)
    y2 = x[0]
    print(y2)


def test_variable_get_item() -> None:
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.get_item(x, 1)
    print(y)

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    indices = np.array([0, 0, 1])
    y = F.get_item(x, indices)
    print(y)

    y = x[1]
    print(y)
    y = x[:, 2]
    print(y)


def check_model_result() -> None:
    model = MLP((10, 3))
    x = np.array([[0.2, -0.4]])
    y = model(x)
    print(y)


def softmax1d(x: ArrayType | Variable) -> Variable:
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y


def test_softmax1d() -> None:
    model = MLP((10, 3))
    x = Variable(np.array([[0.2, -0.4]]))
    y = model(x)
    p = softmax1d(y)
    print(y)
    print(p)


def test_softmax_simple() -> None:
    model = MLP((10, 3))
    x = Variable(np.array([[0.2, -0.4]]))
    y = model(x)
    p = F.softmax_simple(y)
    print(y)
    print(p)


def test_softmax_cross_entropy() -> None:
    model = MLP((10, 3))
    x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
    t = np.array([2, 0, 1, 0])
    y = model(x)
    loss = F.softmax_cross_entropy_simple(y, t)
    loss.backward()
    print(loss)


if __name__ == "__main__":
    test_numpy_get_item()

    test_variable_get_item()

    check_model_result()

    test_softmax1d()

    test_softmax_simple()

    test_softmax_cross_entropy()
