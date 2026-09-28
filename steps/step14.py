from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")

        self.data: np.ndarray = data
        self.grad: np.ndarray | None = None
        self.creator: Function | None = None

    def set_creator(self, creator: Function) -> None:
        self.creator = creator

    def cleargrad(self) -> None:
        self.grad = None

    def backward(self) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        if self.creator is None:
            return None
        funcs: list[Function] = [self.creator]

        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    funcs.append(x.creator)


class Function(ABC):
    inputs: Sequence[Variable]
    output: Sequence[Variable]

    def __call__(self, *inputs: Variable) -> Sequence[Variable] | Variable:
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(np.asarray(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self, *x: np.ndarray) -> Sequence[np.ndarray]:
        ...

    @abstractmethod
    def backward(self, *gy: np.ndarray) -> np.ndarray:
        ...


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x * x
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x: Variable) -> Variable | Sequence[Variable]:
    return Square()(x)


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> Sequence[np.ndarray]:
        y = x0 + x1
        return (y, )

    def backward(self, gy: np.ndarray) -> Sequence[np.ndarray]:
        return gy, gy


def add(x0: Variable, x1: Variable) -> Variable | Sequence[Variable]:
    return Add()(x0, x1)


if __name__ == "__main__":
    x = Variable(np.array(3.0))
    y = add(x, x)
    y.backward()
    print(x.grad)

    x = Variable(np.array(3.0))  # or x.cleargrad()
    y = add(add(x, x), x)
    y.backward()
    print(x.grad)
