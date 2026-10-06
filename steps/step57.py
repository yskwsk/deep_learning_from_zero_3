import numpy as np

from dezero import Variable
import dezero.functions_conv as Fconv


x1 = np.random.rand(1, 3, 7, 7)
col1 = Fconv.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
print(col1.shape)


x2 = np.random.rand(10, 3, 7, 7)  # 10個のデータ
kernel_size = (5, 5)
stride = (1, 1)
pad = (0, 0)
col2 = Fconv.im2col(x2, kernel_size, stride, pad, to_matrix=True)
print(col2.shape)


print(Fconv.pair(1))
print(Fconv.pair((1, 2)))


N, C, H, W = 1, 5, 15, 15
OC, (KH, KW) = 8, (3, 3)
x = Variable(np.random.randn(N, C, H, W))
Weight = np.random.randn(OC, C, KH, KW)
y = Fconv.conv2d_simple(x, Weight, b=None, stride=1, pad=1)
y.backward()
print(y.shape)
assert x.grad is not None
print(x.grad.shape)
