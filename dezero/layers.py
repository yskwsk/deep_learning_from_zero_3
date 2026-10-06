from __future__ import annotations

import os
import types
import weakref
from abc import ABC
from collections.abc import Generator
from typing import Any, Callable, cast
from weakref import ReferenceType

import numpy as np
import numpy.typing as npt

import dezero.functions as F
import dezero.functions_conv as Fconv
from dezero import cuda
from dezero.array_type import ArrayType, Dtype
from dezero.core import Variable, Parameter
from dezero.utils import pair


# =============================================================================
# Layer (base class)
# =============================================================================

def _forward_unimplemented(self, *inputs: Any) -> None:
    raise NotImplementedError(
        f"Module [{type(self).__name__}] is missing the required 'forward' function"
    )


class Layer(ABC):
    forward: Callable[..., Any] = _forward_unimplemented

    def __init__(self) -> None:
        self._params: set[str] = set()
        self.inputs: list[ReferenceType[ArrayType | Variable]] = []
        self.outputs: list[ReferenceType[Variable]] = []

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter | Layer):
            self._params.add(name)
        super().__setattr__(name, value)

    # def __call__(self, *inputs: ArrayType | Variable) -> Any:
    def __call__(self, *inputs: Any) -> Any:
        ys = self.forward(*inputs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [cast(Variable, y) for y in ys]
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def params(self) -> Generator[Parameter]:
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self) -> None:
        for param in self.params():
            param.cleargrad()

    def to_cpu(self) -> None:
        for param in self.params():
            param.to_cpu()

    def to_gpu(self) -> None:
        for param in self.params():
            param.to_gpu()

    def _flatten_params(
        self,
        params_dict: dict[str, Parameter],
        parent_key: str = ""
    ) -> None:
        for name in self._params:
            obj = self.__dict__[name]
            if parent_key:
                key = parent_key + "/" + name
            else:
                key = name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            elif isinstance(obj, Parameter):
                params_dict[key] = obj

    def save_weights(self, path: str) -> None:
        self.to_cpu()

        params_dict: dict[str, Parameter] = {}
        self._flatten_params(params_dict)
        # array_dict = {key: param.data for key, param in params_dict.items() if param is not None}
        array_dict: dict[str, Any] = {}
        for key, param in params_dict.items():
            if param.data is not None:
                array_dict[key] = param.data

        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt):
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path: str) -> None:
        npz = np.load(path)
        params_dict: dict[str, Parameter] = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


# =============================================================================
# Linear / Conv2d / Deconv2d
# =============================================================================

class Linear(Layer):
    def __init__(
        self,
        out_size: int,
        nobias: bool = False,
        dtype: npt.DTypeLike = np.float32,
        in_size: int | None = None
    ) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self) -> None:
        if self.in_size is None:
            raise ValueError("in_size is None")
        in_size = self.in_size
        out_size = self.out_size
        W_data = np.random.randn(in_size, out_size).astype(self.dtype) * np.sqrt(1.0 / in_size)
        self.W.data = W_data

    def forward(self, x: ArrayType | Variable) -> Variable:
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y


class Conv2d(Layer):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        pad: int | tuple[int, int] = 0,
        nobias: bool = False,
        dtype: Dtype = np.float32,
        in_channels: int | None = None
    ) -> None:
        """Two-dimensional convolutional layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name="W")
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name="b")

    def _init_W(self, xp: types.ModuleType = np) -> None:
        if self.in_channels is None:
            raise ValueError("in_channels is None")

        C = self.in_channels
        OC = self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x: ArrayType | Variable) -> Variable:
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = Fconv.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y


class Deconv2d(Layer):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        pad: int | tuple[int, int] = 0,
        nobias: bool = False,
        dtype: Dtype = np.float32,
        in_channels: int | None = None
    ) -> None:
        """Two-dimensional deconvolutional (transposed convolution)layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp: types.ModuleType = np) -> None:
        if self.in_channels is None:
            raise ValueError("in_channels is None")

        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(C, OC, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x: ArrayType | Variable) -> Variable:
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = Fconv.deconv2d(x, self.W, self.b, self.stride, self.pad)
        return y


# =============================================================================
# RNN / LSTM
# =============================================================================
class RNN(Layer):
    def __init__(self, hidden_size, in_size=None):
        """An Elman RNN with tanh.

        Args:
            hidden_size (int): The number of features in the hidden state.
            in_size (int): The number of features in the input. If unspecified
            or `None`, parameter initialization will be deferred until the
            first `__call__(x)` at which time the size will be determined.

        """
        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size)
        self.h2h = Linear(hidden_size, in_size=in_size, nobias=True)
        self.h = None

    def reset_state(self):
        self.h = None

    def forward(self, x):
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h))
        self.h = h_new
        return h_new


class LSTM(Layer):
    def __init__(
        self,
        hidden_size: int,
        in_size: int | None = None
    ) -> None:
        super().__init__()

        self.x2f = Linear(hidden_size, in_size=in_size)
        self.x2i = Linear(hidden_size, in_size=in_size)
        self.x2o = Linear(hidden_size, in_size=in_size)
        self.x2u = Linear(hidden_size, in_size=in_size)
        self.h2f = Linear(hidden_size, in_size=hidden_size, nobias=True)
        self.h2i = Linear(hidden_size, in_size=hidden_size, nobias=True)
        self.h2o = Linear(hidden_size, in_size=hidden_size, nobias=True)
        self.h2u = Linear(hidden_size, in_size=hidden_size, nobias=True)

        self.h: Variable | None = None
        self.c: Variable | None = None

    def reset_state(self) -> None:
        self.h = None
        self.c = None

    def forward(self, x: ArrayType | Variable) -> Variable:
        if self.h is None:
            f = F.sigmoid(self.x2f(x))
            i = F.sigmoid(self.x2i(x))
            o = F.sigmoid(self.x2o(x))
            u = F.tanh(self.x2u(x))
        else:
            f = F.sigmoid(self.x2f(x) + self.h2f(self.h))
            i = F.sigmoid(self.x2i(x) + self.h2i(self.h))
            o = F.sigmoid(self.x2o(x) + self.h2o(self.h))
            u = F.tanh(self.x2u(x) + self.h2u(self.h))

        if self.c is None:
            c_new = (i * u)
        else:
            c_new = (f * self.c) + (i * u)

        h_new = o * F.tanh(c_new)

        self.h = h_new
        self.c = c_new

        return h_new


# =============================================================================
# EmbedID / BatchNorm
# =============================================================================

class EmbedID(Layer):
    def __init__(
        self,
        in_size: int,
        out_size: int
    ) -> None:
        super().__init__()
        self.W = Parameter(np.random.randn(in_size, out_size), name="W")

    def __call__(self, *inputs: int) -> Variable:
        x = inputs[0]
        y = self.W[x]
        return y


class BatchNorm(Layer):
    def __init__(self):
        super().__init__()
        # `.avg_mean` and `.avg_var` are `Parameter` objects, so they will be
        # saved to a file (using `save_weights()`).
        # But they don't need grads, so they're just used as `ndarray`.
        self.avg_mean = Parameter(None, name='avg_mean')
        self.avg_var = Parameter(None, name='avg_var')
        self.gamma = Parameter(None, name='gamma')
        self.beta = Parameter(None, name='beta')

    def _init_params(self, x: ArrayType | Variable) -> None:
        xp = cuda.get_array_module(x)
        D = x.shape[1]
        if self.avg_mean.data is None:
            self.avg_mean.data = xp.zeros(D, dtype=x.dtype)
        if self.avg_var.data is None:
            self.avg_var.data = xp.ones(D, dtype=x.dtype)
        if self.gamma.data is None:
            self.gamma.data = xp.ones(D, dtype=x.dtype)
        if self.beta.data is None:
            self.beta.data = xp.zeros(D, dtype=x.dtype)

    def __call__(self, *inputs: ArrayType | Variable):
        x = inputs[0]
        if self.avg_mean.data is None:
            self._init_params(x)
        return F.batch_nrom(
            x, self.gamma,
            self.beta,
            self.avg_mean.data,
            self.avg_var.data
        )
