from __future__ import annotations

import contextlib
import weakref
from weakref import ReferenceType

from abc import ABC
from collections.abc import Generator, Sequence
from typing import Any, Callable, ClassVar, cast

import numpy as np


class Config:
    enable_backprop: ClassVar[bool] = True


@contextlib.contextmanager
def using_config(name: str, value: Any) -> Generator[None, None, None]:
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad() -> contextlib.AbstractContextManager[None]:
    return using_config("enable_backprop", False)


class Variable:
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
            gys: list[np.ndarray] = []
            for output_ref in f.outputs:
                output = output_ref()
                if (output is not None) and (output.grad is not None):
                    gys.append(output.grad)
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            gxs = cast(tuple[np.ndarray, ...], gxs)

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

    def __add__(self, other) -> Variable:
        return add(self, other)

    def __mul__(self, other) -> Variable:
        return mul(self, other)


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

    def __call__(self, *inputs: Variable) -> Any:
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(np.asarray(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x * x
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x: Variable) -> Variable:
    return Square()(x)


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 + x1
        return y

    def backward(self, gy: np.ndarray) -> Sequence[np.ndarray]:
        return gy, gy


def add(x0: Variable, x1: Variable) -> Variable:
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 * x1
        return y

    def backward(self, gy: np.ndarray) -> Sequence[np.ndarray]:
        x0 = self.inputs[0].data
        x1 = self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0: Variable, x1: Variable) -> Variable:
    return Mul()(x0, x1)


# Variable.__add__ = add
# Variable.__mul__ = mul


if __name__ == "__main__":
    a = Variable(np.array(3.0))
    b = Variable(np.array(2.0))
    c = Variable(np.array(1.0))

    # y = add(mul(a, b), c)
    y = a * b + c
    y.backward()

    print(y)
    print(a.grad)
    print(b.grad)
