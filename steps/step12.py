from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import Any, Callable

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
    output: Sequence[Variable]

    forward: Callable[..., Any] = _forward_unimplemented
    backward: Callable[..., Any] = _backward_unimplemented

    def __call__(self, *inputs: Variable) -> Any:
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


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> Sequence[np.ndarray]:
        y = x0 + x1
        return (y, )

    def backward(self, gy: np.ndarray) -> np.ndarray:
        return np.array(np.nan)


def add(x0: Variable, x1: Variable) -> Variable:
    return Add()(x0, x1)


if __name__ == "__main__":
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    y = add(x0, x1)
    print(y.data)
