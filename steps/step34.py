import numpy as np
import matplotlib.pyplot as plt

import dezero.functions as F
from dezero import Variable


def test() -> None:
    x = Variable(np.array(1.0))
    y = F.sin(x)
    y.backward(create_graph=True)

    for i in range(3):
        if x.grad is None:
            raise ValueError("x.grad is None")
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
        print(x.grad)


if __name__ == "__main__":
    x = Variable(np.linspace(-7, 7, 200))
    y = F.sin(x)
    y.backward(create_graph=True)
    assert y.data is not None

    logs: list[np.ndarray] = [y.data.flatten()]

    for i in range(3):
        if (x.grad is None) or (x.grad.data is None):
            raise ValueError("x.grad is None")

        logs.append(x.grad.data.flatten())
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)

    lables: list[str] = ["y=sin(x)", "y'", "y''", "y'''"]
    for i, v in enumerate(logs):
        if x.data is None:
            raise ValueError("x.data is None")
        plt.plot(x.data, v, label=lables[i])
    plt.legend(loc="lower right")
    plt.show()
