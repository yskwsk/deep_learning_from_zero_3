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


class Function(ABC):
    inputs: Sequence[Variable]
    output: Sequence[Variable]

    @abstractmethod
    def forward(self, x: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        ...

    @abstractmethod
    def backward(self, gy: np.ndarray) -> np.ndarray:
        ...

    def __call__(self, inputs: Sequence[Variable]) -> Sequence[Variable]:
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(np.asarray(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs


class Add(Function):
    def forward(self, x: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        x0, x1 = x
        y = x0 + x1
        return (y, )

    def backward(self, gy: np.ndarray) -> np.ndarray:
        return np.array(np.nan)


if __name__ == "__main__":
    xs = [
        Variable(np.array(2)),
        Variable(np.array(3))
    ]
    f = Add()
    ys = f(xs)
    y = ys[0]
    print(y.data)
