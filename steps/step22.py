from __future__ import annotations

import contextlib
import weakref
from weakref import ReferenceType

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np


class Config:
    enable_backprop: bool = True


@contextlib.contextmanager
def using_config(name: str, value: Any):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprop", False)


class Variable:
    __array_priority__ = 200

    def __init__(self, data: np.ndarray, name: str | None = None) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")

        self.data: np.ndarray = data
        self.name: str | None = name
        self.grad: np.ndarray | None = None
        self.creator: Function | None = None
        self.generation: int = 0

    @property
    def shape(self) -> Sequence[int]:
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

    def set_creator(self, creator: Function) -> None:
        self.creator = creator
        self.generation = creator.generation + 1

    def cleargrad(self) -> None:
        self.grad = None

    def backward(self, retain_grad: bool = False) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs: list[Function] = []
        seen_set: set[Function] = set()

        def add_func(f: Function | None) -> None:
            if (f is not None) and (f not in seen_set):
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None


def as_variable(obj: np.ndarray | Variable) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x) -> np.ndarray | Variable:
    if np.isscalar(x):
        return np.array(x)
    return x


class Function(ABC):
    inputs: Sequence[Variable]
    outputs: Sequence[ReferenceType[Variable]]
    generation: int

    def __call__(self, *inputs: Variable | np.ndarray) -> Sequence[Variable] | Variable:
        _inputs: list[Variable] = [as_variable(x) for x in inputs]

        xs = [x.data for x in _inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(np.asarray(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in _inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = _inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self, *x: np.ndarray) -> Sequence[np.ndarray]:
        ...

    @abstractmethod
    def backward(self, *gy: np.ndarray) -> np.ndarray:
        ...


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 + x1
        return y

    def backward(self, gy: np.ndarray) -> Sequence[np.ndarray]:
        return gy, gy


def add(x0, x1) -> Variable | Sequence[Variable]:
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 * x1
        return y

    def backward(self, gy: np.ndarray) -> Sequence[np.ndarray]:
        x0 = self.inputs[0].data
        x1 = self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0, x1) -> Variable:
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return -x

    def backward(self, gy: np.ndarray) -> np.ndarray:
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
            y = x0 - x1
            return y

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return sub(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return div(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c

        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


Variable.__add__ = add
Variable.__radd__ = add
Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__pow__ = pow


if __name__ == "__main__":
    x = Variable(np.array(2.0))
    y = -x
    print(y)

    y1 = 2.0 - x
    y2 = x - 1.0
    print(y1)
    print(y2)

    y = 3.0 / x
    print(y)

    y = x ** 3
    y.backward()
    print(y)
