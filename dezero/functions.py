from __future__ import annotations

from collections.abc import Sequence
from types import EllipsisType
from typing import Any, Literal, TypeAlias, cast

import numpy as np

import dezero
from dezero import cuda, utils
from dezero.array_type import ArrayType
from dezero.core import Function, Parameter, Variable, as_variable, as_array


# =============================================================================
# Basic functions: sin / cos / tanh / exp / log
# =============================================================================

class Sin(Function):
    def forward(self, x: ArrayType) -> ArrayType:
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x: ArrayType | Variable) -> Variable:
    return Sin()(x)


class Cos(Function):
    def forward(self, x: ArrayType) -> ArrayType:
        y = np.cos(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x: ArrayType | Variable) -> Variable:
    return Cos()(x)


class Tanh(Function):
    def forward(self, x: ArrayType) -> ArrayType:
        y = np.tanh(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        assert y is not None

        gx = gy * (1 - y * y)
        return gx


def tanh(x: ArrayType | Variable) -> Variable:
    return Tanh()(x)


class Exp(Function):
    def forward(self, x: ArrayType) -> ArrayType:
        y = np.exp(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        assert y is not None

        gx = gy * y
        return gx


def exp(x: ArrayType | Variable) -> Variable:
    return Exp()(x)


class Log(Function):
    def forward(self, x: ArrayType) -> ArrayType:
        y = np.log(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        gx = gy / x
        return gx


def log(x: ArrayType | Variable) -> Variable:
    return Log()(x)


# =============================================================================
# Tensor operations: reshape / transpose / get_item / expand_dims / flatten
# =============================================================================

class Reshape(Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape: tuple[int, ...] = shape

        # 元の変数の形状
        self.x_shape: tuple[int, ...] | None = None

    def forward(self, x: ArrayType) -> ArrayType:
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy) -> Variable:
        assert self.x_shape is not None

        # 元の変数と同じ形状に変換する
        return reshape(gy, self.x_shape)


def reshape(
    x: ArrayType | Variable,
    shape: tuple[int, ...]
) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes: tuple[int, ...] | None = None) -> None:
        self.axes = axes

    def forward(self, x: ArrayType) -> ArrayType:
        y = x.transpose(self.axes)
        return y

    def backward(self, gy: Variable) -> Variable:
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))

        return transpose(gy, inv_axes)


def transpose(
    x: ArrayType | Variable,
    axes: tuple[int, ...] | None = None
) -> Variable:
    return Transpose(axes)(x)


IndexPartType: TypeAlias = int | slice | EllipsisType | ArrayType


IndexType: TypeAlias = IndexPartType | tuple[IndexPartType, ...]


class GetItem(Function):
    def __init__(self, slices: IndexType):
        self.slices = slices

    def forward(self, x: ArrayType) -> ArrayType:
        y = x[self.slices]
        return y

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


def get_item(
    x: ArrayType | Variable,
    slices: IndexType
) -> Variable:
    return GetItem(slices)(x)


class GetItemGrad(Function):
    def __init__(
        self,
        slices: IndexType,
        in_shape: tuple[int, ...]
    ) -> None:
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy: ArrayType) -> ArrayType:
        xp = cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)

        if xp is np:
            indices = cast(Any, self.slices)
            np.add.at(gx, indices, gy)
        else:
            xp.scatter_add(gx, self.slices, gy)

        return gx

    def backward(self, ggx: Variable) -> Variable:
        return get_item(ggx, self.slices)


def expand_dims(x: ArrayType | Variable, axis: int) -> Variable:
    x = as_variable(x)
    shape = list(x.shape)
    shape.insert(axis, 1)
    return reshape(x, tuple(shape))


def flatten(x: ArrayType | Variable) -> Variable:
    """Flattens the input. Does not affect the batch size."""
    return reshape(x, (x.shape[0], -1))


# =============================================================================
# sum / sum_to / broadcast_to / average / matmul / linear
# =============================================================================

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

    def forward(self, x: ArrayType) -> ArrayType:
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy: Variable) -> Variable:
        assert self.x_shape is not None

        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(
    x: ArrayType | Variable,
    axis: int | tuple[int, ...] | None = None,
    keepdims: Literal[True, False] = False
) -> Variable:
    return Sum(axis, keepdims)(x)


class SumTo(Function):
    def __init__(self, shape: Sequence[int]) -> None:
        self.shape = shape

        # 元の配列の形状
        self.x_shape: tuple[int, ...] | None = None

    def forward(self, x: ArrayType) -> ArrayType:
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy: Variable) -> Variable:
        assert self.x_shape is not None

        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(
    x: ArrayType | Variable,
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

    def forward(self, x: ArrayType) -> ArrayType:
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy: Variable) -> Variable:
        assert self.x_shape is not None

        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x: ArrayType | Variable, shape: Sequence[int]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class MatMul(Function):
    def forward(self, x: ArrayType, W: ArrayType) -> ArrayType:
        y = x.dot(W)
        return y

    def backward(self, gy: Variable) -> tuple[Variable, Variable]:
        x, W = self.inputs

        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)

        return gx, gW


def matmul(
    x: ArrayType | Variable,
    W: ArrayType | Variable
) -> Variable:
    return MatMul()(x, W)


class Linear(Function):
    def forward(self, x: ArrayType, W: ArrayType, b: ArrayType | None = None) -> ArrayType:
        print(type(x), type(W), type(b))
        y = x.dot(W)

        if b is not None:
            y += b
        return y

    def backward(self, gy: Variable) -> tuple[Variable, Variable, Variable | None]:
        x, W, b = self.inputs
        if b.data is None:
            gb = None
        else:
            gb = sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(
    x: ArrayType | Variable,
    W: ArrayType | Variable,
    b: ArrayType | Variable | None = None
) -> Variable:
    return Linear()(x, W, b)


def linear_simple(
    x: ArrayType | Variable,
    W: ArrayType | Variable,
    b: ArrayType | Variable | None = None
) -> Variable:
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    # Release t.data (ndarray) for memory efficiency
    t.data = None
    return y


# =============================================================================
# activation function: sigmoid / relu / softmax / log_softmax / leaky_relu
# =============================================================================

def sigmoid_simple(x: ArrayType | Variable) -> Variable:
    x = as_variable(x)
    y = 1.0 / (1.0 + exp(-x))
    return y


class Sigmoid(Function):
    def forward(self, x: ArrayType) -> ArrayType:
        y = 1.0 / (1.0 + np.exp(-x))
        # y = np.tanh(0.5 * x) * 0.5 + 0.5
        return y

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        assert y is not None

        gx = gy * y * (1 - y)
        return gx


def sigmoid(x: ArrayType | Variable) -> Variable:
    return Sigmoid()(x)


class ReLU(Function):
    def forward(self, x: ArrayType) -> ArrayType:
        y = np.maximum(x, 0.0)
        return y

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        assert x.data is not None

        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x: ArrayType | Variable) -> Variable:
    return ReLU()(x)


def softmax_simple(
    x: ArrayType | Variable,
    axis: int | tuple[int, ...] | None = 1,
) -> Variable:
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y


class Softmax(Function):
    def __init__(self, axis: int | tuple[int, ...] | None = 1):
        self.axis = axis

    def forward(self, x: ArrayType) -> ArrayType:
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        assert y is not None

        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(
    x: ArrayType | Variable,
    axis: int | tuple[int, ...] | None = 1
) -> Variable:
    return Softmax(axis)(x)


class LogSoftmax(Function):
    def __init__(self, axis: int | tuple[int, ...] | None = 1) -> None:
        self.axis = axis

    def forward(self, x: ArrayType) -> ArrayType:
        log_z = utils.logsumexp(x, self.axis)
        y = x - log_z
        return y

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        assert y is not None

        gx = gy - exp(y) * gy.sum(axis=self.axis, keepdims=True)
        return gx


def log_softmax(
    x: ArrayType | Variable,
    axis: int | tuple[int, ...] | None = 1
) -> Variable:
    return LogSoftmax(axis)(x)


class LeakyReLU(Function):
    def __init__(self, slope: float) -> None:
        self.slope = slope

    def forward(self, x: ArrayType) -> ArrayType:
        y = x.copy()
        y[x <= 0] *= self.slope
        return y

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        assert x.data is not None

        mask = (x.data > 0).astype(gy.dtype)
        mask[mask <= 0] = self.slope
        gx = gy * mask
        return gx


def leaky_relu(
    x: ArrayType | Variable,
    slope: float = 0.2
) -> Variable:
    return LeakyReLU(slope)(x)


# =============================================================================
# loss function: mean_squared_error / softmax_cross_entropy / sigmoid_cross_entropy / binary_cross_entropy
# =============================================================================

def mean_squared_error_simple(
    x0: ArrayType | Variable,
    x1: ArrayType | Variable
) -> Variable:
    x0 = as_variable(x0)
    x1 = as_variable(x1)
    diff = x0 - x1
    y = sum(diff ** 2) / len(diff)
    return y


class MeanSquaredError(Function):
    def forward(self, x0: ArrayType, x1: ArrayType) -> ArrayType:
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
    x0: ArrayType | Variable,
    x1: ArrayType | Variable
) -> Variable:
    return MeanSquaredError()(x0, x1)


def softmax_cross_entropy_simple(
    x: ArrayType | Variable,
    t: ArrayType | Variable
):
    _x = as_variable(x)
    _t = as_variable(t)
    assert _x.data is not None
    assert _t.data is not None

    N = int(_x.shape[0])
    p = softmax(_x)
    # To avoid log(0)
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), _t.data]
    y = -1 * sum(tlog_p) / N

    return y


class SoftmaxCrossEntropy(Function):
    def forward(self, x: ArrayType, t: ArrayType) -> float:
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy: Variable) -> Variable:
        x, t = self.inputs
        assert x.data is not None
        assert t.data is not None

        N, CLS_NUM = x.shape

        gy *= 1 / N
        y = softmax(x)

        # convert to one-hot
        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy

        return y


def softmax_cross_entropy(
    x: ArrayType | Variable,
    t: ArrayType | Variable
) -> Variable:
    return SoftmaxCrossEntropy()(x, t)


def sigmoid_cross_entropy(
    x: ArrayType | Variable,
    t: ArrayType | Variable
) -> Variable:
    if x.ndim != t.ndim:
        t = t.reshape(*x.shape)
    _x = as_variable(x)
    _t = as_variable(t)
    N = len(_x)
    p = sigmoid(_x)
    p = clip(p, 1e-15, 1.0)
    tlog_p = _t * log(p) + (1 - _t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y


def binary_cross_entropy(
    p: ArrayType | Variable,
    t: ArrayType | Variable
) -> Variable:
    if p.ndim != t.ndim:
        t = t.reshape(*p.shape)
    N = len(t)
    p = clip(p, 1e-15, 0.999)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y


# =============================================================================
# accuracy / dropout / batch_norm / embed_id
# =============================================================================

def accuracy(y: ArrayType | Variable, t: ArrayType | Variable) -> Variable:
    """
    [WAR] This function is not differentiable.
    """
    _y = as_variable(y)
    _t = as_variable(t)
    assert _y.data is not None
    assert _t.data is not None

    pred = _y.data.argmax(axis=1).reshape(t.shape)
    result = pred == _t.data
    acc = result.mean()

    return Variable(cast(ArrayType, as_array(acc)))


def dropout(
    x: ArrayType | Variable,
    dropout_ratio: float = 0.5
) -> Variable:
    _x = as_variable(x)

    if dezero.Config.train:
        # Inverted Dropout
        xp = cuda.get_array_module(_x)
        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0 - dropout_ratio).astype(x.dtype)
        y = _x * mask / scale
        return y
    else:
        return _x


class BatchNorm(Function):
    def __init__(
        self,
        mean: ArrayType | Variable | Parameter,
        var: ArrayType | Variable | Parameter,
        decay: float,
        eps: float
    ) -> None:
        self.avg_mean = mean
        self.avg_var = var
        self.decay = decay
        self.eps = eps
        self.inv_std: ArrayType | None = None

    def forward(
        self,
        x: ArrayType,
        gamma: ArrayType,
        beta: ArrayType
    ) -> ArrayType:
        assert x.ndim == 2 or x.ndim == 4
        assert self.inv_std is not None

        x_ndim = x.ndim
        if x_ndim == 4:
            N, C, H, W = x.shape
            # (N, C, H, W) -> (N*H*W, C)
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)

        xp = cuda.get_array_module(x)

        if dezero.Config.train:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            inv_std = 1 / xp.sqrt(var + self.eps)
            xc = (x - mean) * inv_std

            m = x.size // gamma.size
            s = m - 1.0 if m - 1.0 > 1.0 else 1.0
            adjust = m / s  # unbiased estimation
            self.avg_mean *= self.decay
            self.avg_mean += (1 - self.decay) * mean
            self.avg_var *= self.decay
            self.avg_var += (1 - self.decay) * adjust * var
            self.inv_std = inv_std
        else:
            inv_std = 1 / xp.sqrt(self.avg_var + self.eps)
            xc = (x - self.avg_mean) * inv_std
        y = gamma * xc + beta

        if x_ndim == 4:
            # (N*H*W, C) -> (N, C, H, W)
            N, C, H, W = x.shape
            y = y.reshape(N, H, W, C).transpose(0, 3, 1, 2)

        return y

    def backward(self, gy: Variable) -> tuple[Variable, Variable, Variable]:
        assert self.inv_std is not None

        gy_ndim = gy.ndim
        if gy_ndim == 4:
            N, C, H, W = gy.shape
            gy = gy.transpose(0, 2, 3, 1).reshape(-1, C)

        x, gamma, beta = self.inputs
        batch_size = len(gy)

        if x.ndim == 4:
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)

        mean = x.sum(axis=0) / batch_size
        xc = (x - mean) * self.inv_std

        gbeta = sum(gy, axis=0)
        ggamma = sum(xc * gy, axis=0)
        gx = gy - gbeta / batch_size - xc * ggamma / batch_size
        gx *= gamma * self.inv_std

        if gy_ndim == 4:
            N, C, H, W = gy.shape
            gx = gx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

        return gx, ggamma, gbeta


def batch_nrom(
    x: ArrayType | Variable,
    gamma: ArrayType | Variable | Parameter,
    beta: ArrayType | Variable | Parameter,
    mean: ArrayType | Variable | Parameter,
    var: ArrayType | Variable | Parameter,
    decay: float = 0.9,
    eps:  float = 2e-5,
) -> Variable:
    return BatchNorm(mean, var, decay, eps)(x, gamma, beta)


def embed_id(
    x: int,
    W: Variable
) -> Variable:
    return W[x]


# =============================================================================
# max / min / clip
# =============================================================================

class Max(Function):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: Literal[True, False] = False
    ) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: ArrayType) -> ArrayType:
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy: Variable) -> Variable:
        x = self.inputs[0]
        y = self.outputs[0]()
        assert x.data is not None
        assert y is not None
        assert y.data is not None

        shape = tuple(utils.max_backward_shape(x, self.axis))
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        assert y.data is not None
        cond = x.data == y.data
        gy = broadcast_to(gy, cond.shape)
        return gy * cond


def max(
    x: ArrayType | Variable,
    axis: int | tuple[int, ...] | None = None,
    keepdims: Literal[True, False] = False
) -> Variable:
    return Max(axis, keepdims)(x)


class Min(Max):
    def forward(self, x: ArrayType) -> ArrayType:
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y


def min(
    x: ArrayType | Variable,
    axis: int | tuple[int, ...] | None = None,
    keepdims: Literal[True, False] = False
) -> Variable:
    return Min(axis, keepdims)(x)


class Clip(Function):
    def __init__(self, x_min: float, x_max: float) -> None:
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x: ArrayType) -> ArrayType:
        xp = cuda.get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        assert x.data is not None
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(
    x: ArrayType | Variable,
    x_min: float,
    x_max: float
) -> Variable:
    return Clip(x_min, x_max)(x)
