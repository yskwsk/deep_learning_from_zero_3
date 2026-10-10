import numpy as np
import matplotlib.pyplot as plt

import dezero
from dezero import Variable
import dezero.functions as F
import dezero.layers as L
from dezero import Model


class SimpleRNN(Model):
    def __init__(
        self,
        hidden_size: int,
        out_size: int
    ) -> None:
        super().__init__()
        self.rnn = L.RNN(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self) -> None:
        self.rnn.reset_state()

    def __call__(self, *inputs):
        x = inputs[0]
        h = self.rnn(x)
        y = self.fc(h)
        return y


def test_plot_sincurve() -> None:
    train_set = dezero.datasets.SinCurve(train=True)
    print(len(train_set))
    print(train_set[0])
    print(train_set[1])
    print(train_set[2])

    # 図を描画
    xs = [example[0] for example in train_set]
    ts = [example[1] for example in train_set]
    plt.plot(np.arange(len(xs)), xs, label="xs")
    plt.plot(np.arange(len(ts)), ts, label="ts")
    plt.show()


def main() -> None:
    # ハイパーパラメータの設定
    max_epoch = 100
    hidden_size = 100
    bptt_length = 30  # BPTTの長さ

    train_set = dezero.datasets.SinCurve(train=True)
    seqlen = len(train_set)

    model = SimpleRNN(hidden_size, 1)
    optimizer = dezero.optimizers.Adam().setup(model)

    # 学習の開始
    for epoch in range(max_epoch):
        model.reset_state()
        loss = Variable(np.array(0.0))
        count = 0

        for x, t in train_set:
            x = x.reshape(1, 1)  # 形状を(1, 1)へ変換
            y = model(x)
            loss += F.mean_squared_error(y, t)
            count += 1

            # Truncated BPTTのタイミングを調整
            if count % bptt_length == 0 or count == seqlen:
                model.cleargrads()
                loss.backward()
                loss.unchain_backward()
                optimizer.update()

        assert loss.data is not None
        avg_loss = float(loss.data) / count
        print('| epoch %d | loss %f' % (epoch + 1, avg_loss))

    # プロット
    xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
    model.reset_state()  # モデルのリセット
    pred_list = []

    with dezero.no_grad():
        for x in xs:
            x = np.array(x).reshape(1, 1)
            y = model(x)
            print(type(y), y)
            pred_list.append(float(y.data[0][0]))

    plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
    plt.plot(np.arange(len(xs)), pred_list, label='predict')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_plot_sincurve()

    main()
