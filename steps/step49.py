import math

import matplotlib.pyplot as plt
import numpy as np

import dezero
import dezero.functions as F
from dezero import optimizers
from dezero.models import MLP


def test_spiral_dataset() -> None:
    train_set = dezero.datasets.Spiral()

    batch_index = [0, 1, 2]
    batch = [train_set[i] for i in batch_index]
    x = np.array([example[0] for example in batch])
    t = np.array([example[1] for example in batch])

    print(x.shape)
    print(t.shape)


def plot_loss_history(loss_history: list[float]) -> None:
    epochs = np.arange(1, len(loss_history) + 1)
    plt.plot(epochs, loss_history)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()


def main() -> None:
    # ハイパーパラメータの設定
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 1.0

    # データ、モデル、オプティマイザ
    train_set = dezero.datasets.Spiral()
    model = MLP((hidden_size, 3))
    optimizer = optimizers.SGD(lr).setup(model)

    # 学習の実行
    data_size = len(train_set)
    max_iter = math.ceil(data_size / batch_size)
    loss_history: list[float] = []

    for epoch in range(max_epoch):
        # データセットのインデックスのシャッフル
        index = np.random.permutation(data_size)
        sum_loss = 0.0

        for i in range(max_iter):
            # ミニバッチの取り出し
            batch_index = index[i * batch_size:(i + 1) * batch_size]
            batch = [train_set[i] for i in batch_index]
            batch_x = np.array([example[0] for example in batch])
            batch_t = np.array([example[1] for example in batch])

            # 勾配の算出、パラメータの更新
            y = model(batch_x)
            loss = F.softmax_cross_entropy(y, batch_t)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            if loss.data is not None:
                sum_loss += float(loss.data) * len(batch_t)

        # エポック毎に学習経過を出力
        avg_loss = sum_loss / data_size
        loss_history.append(avg_loss)
        print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))

    plot_loss_history(loss_history)

    # 決定境界
    h = 0.001
    x = train_set.get_raw_data()
    t = train_set.get_raw_label()
    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X = np.c_[xx.ravel(), yy.ravel()]

    with dezero.no_grad():
        score = model(X)
    predict_cls = np.argmax(score.data, axis=1)
    Z = predict_cls.reshape(xx.shape)
    plt.contourf(xx, yy, Z)

    # データセットの各点のプロット
    markers = ['o', 'x', '^']
    colors = ['orange', 'blue', 'green']
    for i in range(len(x)):
        c = t[i]
        plt.scatter(x[i, 0], x[i, 1], s=40,  marker=markers[c], c=colors[c])
    plt.show()


if __name__ == "__main__":
    # test_spiral_dataset()
    main()
