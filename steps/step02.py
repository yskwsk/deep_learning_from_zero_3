from abc import ABC, abstractmethod

import numpy as np


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data


class Function(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        ...

    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * x


if __name__ == "__main__":
    x = Variable(np.array(10))
    f = Square()
    y = f(x)
    print(type(y))
    print(y.data)
