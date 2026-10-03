from __future__ import annotations

import weakref
from abc import ABC
from collections.abc import Sequence
from typing import Any, Callable, Literal, TYPE_CHECKING, cast
from weakref import ReferenceType

import numpy as np

import dezero
from dezero.config import Config, using_config
if TYPE_CHECKING:
    import dezero.functions


# try:
#     import cupy  # type: ignore[import-not-found]
#     array_types = (np.ndarray, cupy.ndarray)
# except ImportError:
#     array_types = np.ndarray


class Variable:
    __array_priority__ = 200

    def __init__(
        self,
        data: np.ndarray,
        name: str | None = None
    ) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError('{} is not supported'.format(type(data)))

        self.data: np.ndarray = data
        self.name: str | None = name
        self.grad: Variable | None = None
        self.creator: Function | None = None
        self.generation: int = 0

    @property
    def shape(self) -> tuple[int, ...]:
        if self.data is None:
            raise ValueError("data is None")
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return f"variable({p})"

    def set_creator(self, func: Function) -> None:
        self.creator = func
        self.generation = func.generation + 1

    def unchain(self) -> None:
        self.creator = None

    def cleargrad(self) -> None:
        self.grad = None

    def backward(
        self,
        retain_grad: bool = False,
        create_graph: bool = False
    ):
        if self.grad is None:
            # xp = dezero.cuda.get_array_module(self.data)
            # self.grad = Variable(xp.ones_like(self.data))
            self.grad = Variable(np.ones_like(self.data))

        funcs: list[Function] = []
        seen_set: set[Function] = set()

        def add_func(f: Function):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        if self.creator is not None:
            add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys: list[Variable] = []
            for output_ref in f.outputs:
                output = output_ref()
                if (output is not None) and (output.grad is not None):
                    gys.append(output.grad)

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
                gxs = cast(tuple[Variable], gxs)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y_ref in f.outputs:
                    y = y_ref()
                    if y is not None:
                        y.grad = None

    def unchain_backward(self) -> None:
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()

    def reshape(self, *shape: int | list[int] | tuple[int, ...]) -> Variable:
        if len(shape) == 1:
            if isinstance(shape[0], tuple):
                shape = shape[0]
            elif isinstance(shape[0], list):
                shape = tuple(shape[0])
        shape = cast(tuple[int, ...], shape)
        return dezero.functions.reshape(self, shape)

    def transpose(self, *axes: int | list[int] | tuple[int, ...]):
        if len(axes) == 0:
            return dezero.functions.transpose(self)
        elif len(axes) == 1:
            if isinstance(axes[0], tuple):
                _axes = axes[0]
            elif isinstance(axes[0], list):
                _axes = tuple(axes[0])
            else:
                raise TypeError(f"{type(axes[0])} is not supported")
            return dezero.functions.transpose(self, _axes)
        else:
            axes = cast(tuple[int, ...], axes)
            return dezero.functions.transpose(self, axes)

    @property
    def T(self):
        return dezero.functions.transpose(self)

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: Literal[True, False] = False
    ) -> Variable:
        return dezero.functions.sum(self, axis, keepdims)

    def to_cpu(self):
        # if self.data is not None:
        #     self.data = dezero.cuda.as_numpy(self.data)
        pass

    def to_gpu(self):
        # if self.data is not None:
        #     self.data = dezero.cuda.as_cupy(self.data)
        pass

    def __add__(self, other) -> Variable:
        return add(self, other)

    def __radd__(self, other) -> Variable:
        return add(self, other)

    def __mul__(self, other) -> Variable:
        return mul(self, other)

    def __rmul__(self, other) -> Variable:
        return mul(self, other)

    def __neg__(self) -> Variable:
        return neg(self)

    def __sub__(self, other) -> Variable:
        return sub(self, other)

    def __rsub__(self, other) -> Variable:
        return rsub(self, other)

    def __truediv__(self, other) -> Variable:
        return div(self, other)

    def __rtruediv__(self, other) -> Variable:
        return rdiv(self, other)

    def __pow__(self, other) -> Variable:
        return pow(self, other)


class Parameter(Variable):
    pass


def as_variable(obj: np.ndarray | Variable) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x


def _forward_unimplemented(self, *inputs: Any) -> None:
    raise NotImplementedError(
        f"Module [{type(self).__name__}] is missing the required 'forward' function"
    )


def _backward_unimplemented(self, *gys: Any) -> None:
    raise NotImplementedError(
        f"Module [{type(self).__name__}] is missing the required 'backward' function"
    )


class Function(ABC):
    inputs: Sequence[Variable]
    outputs: Sequence[ReferenceType[Variable]]
    generation: int

    forward: Callable[..., Any] = _forward_unimplemented
    backward: Callable[..., Any] = _backward_unimplemented

    def __call__(self, *inputs: np.ndarray | Variable | None) -> Any:
        _inputs = [as_variable(x) for x in inputs if x is not None]

        xs = [x.data for x in _inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in _inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = _inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]


class Add(Function):
    x0_shape: tuple[int, ...]
    x1_shape: tuple[int, ...]

    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        y = x0 + x1
        return y

    def backward(self, gy: Variable):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            # for broadcaset
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x0, x1) -> Variable:
    # x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 * x1
        return y

    def backward(self, gy: Variable) -> tuple[Variable, ...]:
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:
            # for broadcast
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0, x1) -> Variable:
    # x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return -x

    def backward(self, gy: Variable) -> Variable:
        return -gy


def neg(x) -> Variable:
    return Neg()(x)


class Sub(Function):
    x0_shape: tuple[int, ...]
    x1_shape: tuple[int, ...]

    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        y = x0 - x1
        return y

    def backward(self, gy) -> tuple[Variable, ...]:
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:
            # for broadcast
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0, x1) -> Variable:
    # x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1) -> Variable:
    # x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 / x1
        return y

    def backward(self, gy: Variable) -> tuple[Variable, ...]:
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:
            # for broadcast
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def div(x0, x1) -> Variable:
    # x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1) -> Variable:
    # x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x ** self.c
        return y

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c) -> Variable:
    return Pow(c)(x)


def setup_variable():
    pass
    # Variable.__getitem__ = dezero.functions.get_item

    # Variable.matmul = dezero.functions.matmul
    # Variable.dot = dezero.functions.matmul
    # Variable.max = dezero.functions.max
    # Variable.min = dezero.functions.min
