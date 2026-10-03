from __future__ import annotations

import numpy as np

from dezero.core import Function, Variable, as_variable


class Sin(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.sin(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x: np.ndarray | Variable) -> Variable:
    return Sin()(x)


class Cos(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.cos(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x: np.ndarray | Variable) -> Variable:
    return Cos()(x)


class Tanh(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.tanh(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        if y is None:
            raise ValueError("y is None")
        gx = gy * (1 - y * y)
        return gx


def tanh(x: np.ndarray | Variable) -> Variable:
    return Tanh()(x)


class Reshape(Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape: tuple[int, ...] = shape

        # 元の変数の形状
        self.x_shape: tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy) -> Variable:
        # 元の変数と同じ形状に変換する
        if self.x_shape is None:
            raise ValueError("x_shape is None")
        return reshape(gy, self.x_shape)


def reshape(x: np.ndarray | Variable, shape: tuple[int, ...]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)
