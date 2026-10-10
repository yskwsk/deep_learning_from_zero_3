import os

import numpy as np

import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP


def test_np_save_and_load1() -> None:
    x = np.array([1, 2, 3])
    np.save("test.npy", x)

    y = np.load("test.npy")
    print(y)


def test_np_save_and_load2() -> None:
    x1 = np.array([1, 2, 3])
    x2 = np.array([4, 5, 6])

    np.savez("test.npz", x1=x1, x2=x2)

    arrays = np.load("test.npz")
    x1 = arrays["x1"]
    x2 = arrays["x2"]
    print(x1)
    print(x2)


def test_np_save_and_load3() -> None:
    x1 = np.array([1, 2, 3])
    x2 = np.array([4, 5, 6])
    data = {
        "x1": x1,
        "x2": x2,
    }

    np.savez("test.npz", allow_pickle=False, **data)

    arrays = np.load("test.npz")
    x1 = arrays["x1"]
    x2 = arrays["x2"]
    print(x1)
    print(x2)


def main() -> None:
    max_epoch = 3
    batch_size = 100

    train_set = dezero.datasets.MNIST(train=True)
    train_loader = DataLoader(train_set, batch_size)
    model = MLP((1000, 10))
    optimizer = optimizers.SGD().setup(model)

    if os.path.exists('my_mlp.npz'):
        print("load model: my_mlp.npz")
        model.load_weights('my_mlp.npz')

    for epoch in range(max_epoch):
        sum_loss = 0.0

        for x, t in train_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            assert loss.data is not None
            sum_loss += float(loss.data) * len(t)

        print('epoch: {}, loss: {:.4f}'.format(
            epoch + 1, sum_loss / len(train_set)))

    model.save_weights('my_mlp.npz')


if __name__ == "__main__":
    test_np_save_and_load1()

    test_np_save_and_load2()

    test_np_save_and_load3()

    main()
