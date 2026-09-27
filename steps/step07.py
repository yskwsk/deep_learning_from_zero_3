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
        # 関数の取得
        f = self.creator
        if f is not None:
            # 関数の入力の取得
            x = f.input
            if self.grad is None:
                raise ValueError("grad is None")
            # 関数のbackwardメソッドを呼ぶ
            x.grad = f.backward(self.grad)
            # 自分より１つ前の変数のbackwardメソッドを呼ぶ(再帰)
            x.backward()


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


def test() -> None:
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    assert y.creator == C
    if y.creator is None:
        raise ValueError("y.creator is None")
    assert y.creator.input == b
    assert y.creator.input.creator == B
    if y.creator.input.creator is None:
        raise ValueError("y.creator.input.creator is None")
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == A
    if y.creator.input.creator.input.creator is None:
        raise ValueError("y.creator.input.creator.input.creator is None")
    assert y.creator.input.creator.input.creator.input == x

    y.grad = np.array(1.0)

    _C = y.creator
    _b = _C.input
    _b.grad = _C.backward(y.grad)

    _B = _b.creator
    if _B is None:
        raise ValueError("_B is None")
    _a = _B.input
    _a.grad = _B.backward(_b.grad)

    _A = _a.creator
    if _A is None:
        raise ValueError("_A is None")
    _x = _A.input
    _x.grad = _A.backward(_a.grad)
    print(_x.grad)


if __name__ == "__main__":
    test()

    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    # 逆伝播
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
