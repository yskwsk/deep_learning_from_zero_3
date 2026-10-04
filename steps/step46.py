import matplotlib.pyplot as plt
import numpy as np

from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP


np.random.seed(0)

# データセットの生成
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# ハイパーパラメータの設定
lr = 0.2
max_iter = 10000
hidden_size = 10


model = MLP((hidden_size, 1))
optimizer = optimizers.SGD(lr)
optimizer.setup(model)

# 学習の開始
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()

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
