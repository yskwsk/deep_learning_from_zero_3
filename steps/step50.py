from collections.abc import Iterator
from typing import Self

import matplotlib.pyplot as plt
import numpy as np

import dezero
import dezero.functions as F
from dezero import DataLoader
from dezero import optimizers
from dezero.models import MLP


def test_iterator1() -> None:
    x = [1, 2, 3]
    it = iter(x)
    print(next(it))
    print(next(it))
    print(next(it))
    # print(next(it))


class MyIterator(Iterator[int]):
    def __init__(self, max_cnt: int) -> None:
        self.max_cnt = max_cnt
        self.cnt = 0

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> int:
        if self.cnt == self.max_cnt:
            raise StopIteration()

        self.cnt += 1
        return self.cnt


def test_my_iterator() -> None:
    obj = MyIterator(max_cnt=5)
    for x in obj:
        print(x)


def test_data_loader() -> None:
    batch_size = 10
    max_epoch = 1

    train_set = dezero.datasets.Spiral(train=True)
    test_set = dezero.datasets.Spiral(train=False)
    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, batch_size)

    for epoch in range(max_epoch):
        for x, t in train_loader:
            print(x.shape, t.shape)
            break

        for x, t in test_loader:
            print(x.shape, t.shape)
            break


def main() -> None:
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 1.0

    train_set = dezero.datasets.Spiral(train=True)
    test_set = dezero.datasets.Spiral(train=False)
    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    model = MLP((hidden_size, 3))
    optimizer = optimizers.SGD(lr).setup(model)

    train_loss_history: list[float] = []
    test_loss_history: list[float] = []
    train_accuracy_history: list[float] = []
    test_accuracy_history: list[float] = []

    for epoch in range(max_epoch):
        sum_loss = 0.0
        sum_acc = 0.0

        for x, t in train_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            assert loss.data is not None
            assert acc.data is not None
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

        train_loss = sum_loss / len(train_set)
        train_acc = sum_acc / len(train_set)
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_acc)
        print('epoch: {}'.format(epoch+1))
        print('train loss: {:.4f}, accuracy: {:.4f}'.format(train_loss, train_acc))

        sum_loss = 0.0
        sum_acc = 0.0
        with dezero.no_grad():
            for x, t in test_loader:
                y = model(x)
                loss = F.softmax_cross_entropy(y, t)
                acc = F.accuracy(y, t)

                assert loss.data is not None
                assert acc.data is not None
                sum_loss += float(loss.data) * len(t)
                sum_acc += float(acc.data) * len(t)

        test_loss = sum_loss / len(test_set)
        test_acc = sum_acc / len(test_set)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_acc)
        print('test loss: {:.4f}, accuracy: {:.4f}'.format(test_loss, test_acc))

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4)
    )

    epochs = np.arange(1, max_epoch + 1)
    axes[0].plot(epochs, train_loss_history)
    axes[0].plot(epochs, test_loss_history)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")

    axes[1].plot(epochs, train_accuracy_history)
    axes[1].plot(epochs, test_accuracy_history)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")

    plt.show()


if __name__ == "__main__":
    # test_iterator1()

    # test_my_iterator()

    # test_data_loader()

    main()
