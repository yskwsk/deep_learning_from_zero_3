import numpy as np
import matplotlib.pyplot as plt

from dezero import Variable, Parameter
import dezero.functions as F
import dezero.layers as L


def test_variable_and_parameter() -> None:
    x = Variable(np.array(1.0))
    p = Parameter(np.array(2.0))
    y1 = x * p
    y2 = p * x

    print(isinstance(x, Parameter))
    print(isinstance(p, Parameter))
    print(isinstance(y1, Parameter))
    print(isinstance(y2, Parameter))


def test_layer() -> None:
    # __setattr__の確認
    layer = L.Layer()

    layer.p1 = Parameter(np.array(1))
    layer.p2 = Parameter(np.array(2))
    layer.p3 = Variable(np.array(3))
    layer.p4 = "test"

    print(layer._params)
    print("-------------")

    for name in layer._params:
        print(name, layer.__dict__[name])


if __name__ == "__main__":
    # test_variable_and_parameter()
    # test_layer()

    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    l1 = L.Linear(10)
    l2 = L.Linear(1)

    def predict(x: np.ndarray | Variable) -> Variable:
        y = l1(x)
        y = F.sigmoid(y)
        y = l2(y)
        return y

    lr = 0.2
    iters = 10000

    for i in range(iters):
        y_pred = predict(x)
        loss = F.mean_squared_error(y, y_pred)

        l1.cleargrads()
        l2.cleargrads()
        loss.backward()

        for layer in [l1, l2]:
            for p in layer.params():
                if (p.data is None) or (p.grad is None) or (p.grad.data is None):
                    raise ValueError("p.data or p.grad is None")
                p.data -= lr * p.grad.data

        if i % 1000 == 0:
            print(loss)

    # Plot
    t = np.arange(0, 1, 0.01)[:, np.newaxis]
    y_pred = predict(t)
    if y_pred.data is None:
        raise ValueError("y_pred.data is None")

    plt.scatter(x, y, s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(t, y_pred.data, color='r')
    plt.show()
