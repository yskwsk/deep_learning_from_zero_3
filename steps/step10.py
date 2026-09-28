from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import unittest


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

    def backward(self) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

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
        output = Variable(np.asarray(y))
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


def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)


def numerical_diff(
    f: Function | Callable[[Variable], Variable],
    x: Variable,
    eps: float = 1e-4
) -> np.ndarray:
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        # ランダムな入力値を生成
        np.random.seed(1234)
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        if x.grad is None:
            raise ValueError("x.grad is None")
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)
