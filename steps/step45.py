import matplotlib.pyplot as plt
import numpy as np

from dezero import Model
from dezero import Variable
import dezero.functions as F
import dezero.layers as L


np.random.seed(0)

# データセットの生成
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# ハイパーパラメータの設定
lr = 0.2
max_iter = 10000
hidden_size = 10


# モデルの定義
class TwoLayerNet(Model):
    def __init__(self, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x: Variable) -> Variable:
        y = self.l1(x)
        y = F.sigmoid(y)
        y = self.l2(y)
        return y


model = TwoLayerNet(hidden_size=hidden_size, out_size=1)

# 学習の開始
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        if (p.data is None) or (p.grad is None) or (p.grad.data is None):
            raise ValueError("p.data or p.grad is None")
        p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(loss)

# Plot
t = np.arange(0, 1, 0.01)[:, np.newaxis]
y_pred = model(t)
if y_pred.data is None:
    raise ValueError("y_pred.data is None")

plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(t, y_pred.data, color='r')
plt.show()
