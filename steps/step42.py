import matplotlib.pyplot as plt
import numpy as np


from dezero import Variable
import dezero.functions as F


def create_dataset(seed: int = 1234, plot: bool = False) -> tuple[Variable, ...]:
    np.random.seed(seed=seed)
    _x = np.random.rand(100, 1)
    _y = 5 + 2 * _x + np.random.rand(100, 1)
    x = Variable(_x)
    y = Variable(_y)

    if plot:
        assert x.data is not None
        assert y.data is not None

        plt.scatter(x.data, y.data, s=10)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    return x, y


def predict(x: Variable, W: Variable, b: Variable) -> Variable:
    y = F.matmul(x, W) + b
    return y


def mean_squared_error(x0: Variable, x1: Variable) -> Variable:
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)


if __name__ == "__main__":
    # データセットの生成
    x, y = create_dataset(plot=False)

    # パラメータ
    W = Variable(np.zeros((1, 1)))
    b = Variable(np.zeros(1))

    # 学習率と反復回数
    lr = 0.1
    iters = 100

    for i in range(iters):
        y_pred = predict(x, W, b)
        loss = mean_squared_error(y, y_pred)

        W.cleargrad()
        b.cleargrad()
        loss.backward()

        # Update .data attribute (No need grads when updating params)
        if (W.data is None) or (W.grad is None) or (W.grad.data is None):
            raise ValueError("W.data or W.grad is None")
        if (b.data is None) or (b.grad is None) or (b.grad.data is None):
            raise ValueError("b.data or b.grad is None")

        W.data -= lr * W.grad.data
        b.data -= lr * b.grad.data
        print(W, b, loss)

    # Plot
    y_pred = predict(x, W, b)
    assert x.data is not None
    assert y.data is not None
    assert y_pred.data is not None
    plt.scatter(x.data, y.data, s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x.data, y_pred.data, color='r')
    plt.show()
