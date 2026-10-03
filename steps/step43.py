import matplotlib.pyplot as plt
import numpy as np


from dezero import Variable
import dezero.functions as F

if __name__ == "__main__":
    np.random.seed(0)

    # データセットの生成
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    # パラメータ
    input_size = 1
    hidden_size = 10
    output_size = 1
    W1 = Variable(0.01 * np.random.randn(input_size, hidden_size))
    b1 = Variable(np.zeros(hidden_size))
    W2 = Variable(0.01 * np.random.randn(hidden_size, output_size))
    b2 = Variable(np.zeros(output_size))

    def predict(x):
        y = F.linear(x, W1, b1)
        y = F.sigmoid(y)
        y = F.linear(y, W2, b2)
        return y

    # 学習率と反復回数
    lr = 0.2
    iters = 10000

    for i in range(iters):
        y_pred = predict(x)
        loss = F.mean_squared_error(y, y_pred)

        W1.cleargrad()
        b1.cleargrad()
        W2.cleargrad()
        b2.cleargrad()
        loss.backward()

        if (W1.grad is None) or (b1.grad is None) or (W2.grad is None) or (b2.grad is None):
            raise ValueError("parameter.grad is None")

        W1.data -= lr * W1.grad.data
        b1.data -= lr * b1.grad.data
        W2.data -= lr * W2.grad.data
        b2.data -= lr * b2.grad.data
        if i % 1000 == 0:
            print(loss)

    # Plot
    t = np.arange(0, 1, 0.01)[:, np.newaxis]
    y_pred = predict(t)

    # plt.scatter(x.data, y.data, s=10)
    plt.scatter(x, y, s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(t, y_pred.data, color='r')
    plt.show()
