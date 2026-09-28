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
        self.generation: int = 0

    def set_creator(self, creator: Function) -> None:
        self.creator = creator
        self.generation = creator.generation + 1

    def cleargrad(self) -> None:
        self.grad = None

    def backward(self) -> None:
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
                    add_func(x.creator)


class Function(ABC):
    inputs: Sequence[Variable]
    output: Sequence[Variable]
    generation: int

    def __call__(self, *inputs: Variable) -> Sequence[Variable] | Variable:
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(np.asarray(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])
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
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()

    print(y.data)
    print(x.grad)
