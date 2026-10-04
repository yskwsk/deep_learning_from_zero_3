import matplotlib.pyplot as plt
import numpy as np

import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP


def test_mnist_dataset() -> None:
    train_set = dezero.datasets.MNIST(train=True, transform=None)
    test_set = dezero.datasets.MNIST(train=False, transform=None)

    print(len(train_set))
    print(len(test_set))

    x, t = train_set[0]
    print(type(x), x.shape)
    print(t)

    plt.imshow(x.reshape(28, 28), cmap="gray")
    plt.axis("off")
    plt.show()
    print(f"label: {t}")


def main() -> None:
    max_epoch = 5
    batch_size = 100
    hidden_size = 1000

    train_set = dezero.datasets.MNIST(train=True)
    test_set = dezero.datasets.MNIST(train=False)
    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    # model = MLP((hidden_size, 10))
    # optimizer = optimizers.SGD().setup(model)
    model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
    optimizer = optimizers.Adam().setup(model)

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
    # test_mnist_dataset()

    main()
