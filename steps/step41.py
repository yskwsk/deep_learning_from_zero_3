import numpy as np

from dezero import Variable
import dezero.functions as F


def test_numpy_dot() -> None:
    # ベクトルの内積
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.dot(a, b)
    print(c)

    # 行列の積
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    c = np.dot(a, b)
    print(c)


if __name__ == "__main__":
    test_numpy_dot()

    np.random.seed(1234)

    x = Variable(np.random.randn(2, 3))
    W = Variable(np.random.randn(3, 4))
    y = F.matmul(x, W)
    y.backward()

    print(y)
    if (x.grad is None) or (W.grad is None):
        raise ValueError("x.grad or W.grad is None")
    print(x.grad.shape)
    print(W.grad.shape)
