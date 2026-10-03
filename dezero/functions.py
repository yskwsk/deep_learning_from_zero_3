from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from dezero import utils
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


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.exp(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        if y is None:
            raise ValueError("y is None")
        gx = gy * y
        return gx


def exp(x: np.ndarray | Variable) -> Variable:
    return Exp()(x)


class Log(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.log(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        gx = gy / x
        return gx


def log(x: np.ndarray | Variable) -> Variable:
    return Log()(x)


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


def reshape(
    x: np.ndarray | Variable,
    shape: tuple[int, ...]
) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes: tuple[int, ...] | None = None) -> None:
        self.axes = axes

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x.transpose(self.axes)
        return y

    def backward(self, gy: Variable) -> Variable:
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


def transpose(x, axes: tuple[int, ...] | None = None) -> Variable:
    return Transpose(axes)(x)


class Sum(Function):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None,
        keepdims: Literal[True, False]
    ) -> None:
        self.axis = axis
        self.keepdims = keepdims

        # 元の変数の形状
        self.x_shape: tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy: Variable) -> Variable:
        if self.x_shape is None:
            raise ValueError("x_shape is None")
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(
    x: np.ndarray | Variable,
    axis: int | tuple[int, ...] | None = None,
    keepdims: Literal[True, False] = False
) -> Variable:
    return Sum(axis, keepdims)(x)


class SumTo(Function):
    def __init__(self, shape: Sequence[int]) -> None:
        self.shape = shape

        # 元の配列の形状
        self.x_shape: tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy: Variable) -> Variable:
        if self.x_shape is None:
            raise ValueError("x_shape is None")
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(
    x: np.ndarray | Variable,
    shape: Sequence[int]
) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class BroadcastTo(Function):
    def __init__(self, shape: Sequence[int]) -> None:
        self.shape = shape

        # 元の配列の形状
        self.x_shape: tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy: Variable) -> Variable:
        if self.x_shape is None:
            raise ValueError("x_shape is None")
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x: np.ndarray | Variable, shape: Sequence[int]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class MatMul(Function):
    def forward(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        y = x.dot(W)
        return y

    def backward(self, gy: Variable) -> tuple[Variable, ...]:
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(
    x: np.ndarray | Variable,
    W: np.ndarray | Variable
) -> Variable:
    return MatMul()(x, W)


class Linear(Function):
    def forward(self, x: np.ndarray, W: np.ndarray, b: np.ndarray | None) -> np.ndarray:
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy: Variable) -> tuple[Variable | None, ...]:
        x, W, b = self.inputs
        if b.data is None:
            gb = None
        else:
            gb = sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(
    x: np.ndarray | Variable,
    W: np.ndarray | Variable,
    b: np.ndarray | Variable | None = None
) -> Variable:
    return Linear()(x, W, b)


def linear_simple(
    x: np.ndarray | Variable,
    W: np.ndarray | Variable,
    b: np.ndarray | Variable | None = None
) -> Variable:
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    # Release t.data (ndarray) for memory efficiency
    t.data = None  # type: ignore[assignment]
    return y


def sigmoid_simple(x: np.ndarray | Variable) -> Variable:
    x = as_variable(x)
    y = 1.0 / (1.0 + exp(-x))
    return y


class Sigmoid(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = 1.0 / (1.0 + np.exp(-x))
        # y = np.tanh(0.5 * x) * 0.5 + 0.5
        return y

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        if y is None:
            raise ValueError("y is None")
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x: np.ndarray | Variable) -> Variable:
    return Sigmoid()(x)


def mean_squared_error_simple(
    x0: np.ndarray | Variable,
    x1: np.ndarray | Variable
) -> Variable:
    x0 = as_variable(x0)
    x1 = as_variable(x1)
    diff = x0 - x1
    y = sum(diff ** 2) / len(diff)
    return y


class MeanSquaredError(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy: Variable) -> tuple[Variable, ...]:
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2.0 / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(
    x0: np.ndarray | Variable,
    x1: np.ndarray | Variable
) -> Variable:
    return MeanSquaredError()(x0, x1)
