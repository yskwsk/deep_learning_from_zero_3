from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        self.data: np.ndarray = data
        self.grad: np.ndarray | None = None
        self.creator: Function | None = None

    def set_creator(self, creator: Function) -> None:
        self.creator = creator

    def backward(self) -> None:
        if self.creator is None:
            return None
        funcs: list[Function] = [self.creator]
        while funcs:
            # 関数を取得
            f = funcs.pop()
            # 関数の入出力を取得
            x = f.input
            y = f.output
            # backwardメソッドを呼ぶ
            if y.grad is None:
                raise ValueError("y.grad is None")
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                # 1つ前の関数をリストに追加
                funcs.append(x.creator)


class Function(ABC):
    input: Variable
    output: Variable

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def backward(self, gy: np.ndarray) -> np.ndarray:
        ...

    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * x

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
