from __future__ import annotations

import weakref
from abc import ABC
from collections.abc import Sequence
from typing import Any, Callable, cast
from weakref import ReferenceType

import numpy as np

from dezero.config import Config


class Variable:
    __array_priority__ = 200

    def __init__(
        self,
        data: np.ndarray,
        name: str | None = None
    ) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError('{} is not supported'.format(type(data)))

        self.data: np.ndarray = data
        self.name: str | None = name
        self.grad: np.ndarray | None = None
        self.creator: Function | None = None
        self.generation: int = 0

    @property
    def shape(self) -> tuple[int, ...]:
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

    def set_creator(self, func: Function) -> None:
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self) -> None:
        self.grad = None

    def backward(self, retain_grad: bool = False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs: list[Function] = []
        seen_set: set[Function] = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        if self.creator is not None:
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

    def __radd__(self, other) -> Variable:
        return add(self, other)

    def __mul__(self, other) -> Variable:
        return mul(self, other)

    def __rmul__(self, other) -> Variable:
        return mul(self, other)

    def __neg__(self) -> Variable:
        return neg(self)

    def __sub__(self, other) -> Variable:
        return sub(self, other)

    def __rsub__(self, other) -> Variable:
        return rsub(self, other)

    def __truediv__(self, other) -> Variable:
        return div(self, other)

    def __rtruediv__(self, other) -> Variable:
        return rdiv(self, other)

    def __pow__(self, other) -> Variable:
        return pow(self, other)


def as_variable(obj: np.ndarray | Variable) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x: Any) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


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

    def __call__(self, *inputs: np.ndarray | Variable) -> Any:
        _inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in _inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in _inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = _inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 + x1
        return y

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, ...]:
        return gy, gy


def add(x0, x1) -> Variable:
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 * x1
        return y

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, ...]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0, x1) -> Variable:
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return -x

    def backward(self, gy: np.ndarray) -> np.ndarray:
        return -gy


def neg(x) -> Variable:
    return Neg()(x)


class Sub(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 - x1
        return y

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, ...]:
        return gy, -gy


def sub(x0, x1) -> Variable:
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1) -> Variable:
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 / x1
        return y

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, ...]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


def div(x0, x1) -> Variable:
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1) -> Variable:
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c) -> None:
        self.c = c

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x ** self.c
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        c = self.c

        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c) -> Variable:
    return Pow(c)(x)


# from __future__ import annotations

# import contextlib
# import numpy as np
# import weakref
# from abc import ABC
# from collections.abc import Generator, Sequence
# from typing import Any, Callable
# from weakref import ReferenceType


# class Config:
#     enable_backprop: bool = True


# @contextlib.contextmanager
# def using_config(name: str, value: Any) -> Generator[None, None, None]:
#     old_value = getattr(Config, name)
#     setattr(Config, name, value)
#     try:
#         yield
#     finally:
#         setattr(Config, name, old_value)


# def no_grad() -> contextlib.AbstractContextManager[None]:
#     return using_config("enable_backprop", False)


# class Variable:
#     __array_priority__ = 200

#     def __init__(self, data: np.ndarray, name: str | None = None) -> None:
#         if data is not None:
#             if not isinstance(data, np.ndarray):
#                 raise TypeError('{} is not supported'.format(type(data)))

#         self.data: np.ndarray = data if data is not None else np.empty(0)
#         # self.data: np.ndarray | None = data
#         self.name: str | None = name
#         self.grad: np.ndarray | None = None
#         self.creator: Function | None = None
#         self.generation: int = 0

#     @property
#     def shape(self) -> tuple[int, ...]:
#         return self.data.shape

#     @property
#     def ndim(self) -> int:
#         return self.data.ndim

#     @property
#     def size(self) -> int:
#         return self.data.size

#     @property
#     def dtype(self) -> np.dtype:
#         return self.data.dtype

#     def __len__(self):
#         return len(self.data)

#     def __repr__(self) -> str:
#         if self.data is None:
#             return "variable is None"
#         p = str(self.data).replace("\n", "\n" + " " * 9)
#         return f"variable({p})"

#     def set_creator(self, func: Function) -> None:
#         self.creator = func
#         self.generation = func.generation + 1

#     def cleangrad(self) -> None:
#         self.grad = None

#     def backward(self, retain_grad: bool = False) -> None:
#         if self.grad is None:
#             self.grad = np.ones_like(self.data)

#         funcs: list[Function] = []
#         seen_set: set[Function] = set()

#         def add_func(f: Function | None) -> None:
#             if (f is not None) and (f not in seen_set):
#                 funcs.append(f)
#                 seen_set.add(f)
#                 funcs.sort(key=lambda x: x.generation)

#         add_func(self.creator)

#         while funcs:
#             f: Function = funcs.pop()
#             gys: list[np.ndarray] = []
#             for output in f.outputs:
#                 _output = output()
#                 assert _output is not None
#                 assert _output.grad is not None
#                 gys.append(_output.grad)
#             gxs = f.backward(*gys)
#             if not isinstance(gxs, tuple):
#                 gxs = (gxs,)

#             for x, gx in zip(f.inputs, gxs):
#                 if x.grad is None:
#                     x.grad = gx
#                 else:
#                     x.grad = x.grad + gx

#                 if x.creator is not None:
#                     add_func(x.creator)

#             if not retain_grad:
#                 for y in f.outputs:
#                     _y = y()
#                     assert _y is not None
#                     _y.grad = None

#     def __add__(self, other) -> Variable:
#         return add(self, other)

#     def __radd__(self, other) -> Variable:
#         return add(self, other)

#     def __mul__(self, other) -> Variable:
#         return mul(self, other)

#     def __rmul__(self, other) -> Variable:
#         return mul(self, other)

#     def __neg__(self) -> Variable:
#         return neg(self)

#     def __sub__(self, other) -> Variable:
#         return sub(self, other)

#     def __rsub__(self, other) -> Variable:
#         return rsub(self, other)

#     def __truediv__(self, other) -> Variable:
#         return div(self, other)

#     def __rtruediv__(self, other) -> Variable:
#         return rdiv(self, other)

#     def __pow__(self, other) -> Variable:
#         return pow(self, other)

#     def __rpow__(self, other) -> Variable:
#         return rpow(self, other)


# def as_variable(obj: np.ndarray | Variable) -> Variable:
#     if isinstance(obj, Variable):
#         return obj
#     return Variable(obj)


# # def as_array(x: Any):
# #     if np.isscalar(x):
# #         return np.array(x)
# #     return x
# def as_array(x: Any) -> np.ndarray | Variable:
#     if isinstance(x, Variable):
#         return x
#     return np.asarray(x)


# def _forward_unimplemented(self, *inputs: Any) -> None:
#     raise NotImplementedError(
#         f"Module [{type(self).__name__}] is missing the required 'forward' function"
#     )


# def _backward_unimplemented(self, *gys: Any) -> None:
#     raise NotImplementedError(
#         f"Module [{type(self).__name__}] is missing the required 'backward' function"
#     )


# class Function(ABC):
#     generation: int
#     inputs: Sequence[Variable]
#     outputs: Sequence[ReferenceType[Variable]]

#     def __call__(self, *inputs: np.ndarray | Variable) -> Any:
#         _inputs: list[Variable] = [as_variable(x) for x in inputs]

#         xs: list[np.ndarray] = [x.data for x in _inputs if x.data is not None]
#         ys = self.forward(*xs)
#         if not isinstance(ys, tuple):
#             ys = (ys,)
#         outputs: list[Variable] = [Variable(np.asarray(y)) for y in ys]

#         if Config.enable_backprop:
#             self.generation = max([x.generation for x in _inputs])
#             for output in outputs:
#                 output.set_creator(self)
#             self.inputs = _inputs
#             self.outputs = [weakref.ref(output) for output in outputs]

#         return outputs if len(outputs) > 1 else outputs[0]

#     # @abstractmethod
#     # def forward(self, *xs: Any) -> np.ndarray | tuple[np.ndarray]:
#     #     ...
#     forward: Callable[..., Any] = _forward_unimplemented

#     # @abstractmethod
#     # def _backward(self, *gys: np.ndarray) -> np.ndarray | tuple[np.ndarray]:
#     #     ...
#     backward: Callable[..., Any] = _backward_unimplemented


# class Add(Function):
#     def forward(self, x0, x1):
#         y = x0 + x1
#         return y

#     def backward(self, gy):
#         return gy, gy


# def add(x0, x1):
#     x1 = as_array(x1)
#     return Add()(x0, x1)


# class Mul(Function):
#     def forward(self, x0, x1):
#         y = x0 * x1
#         return y

#     def backward(self, gy):
#         x0 = self.inputs[0].data
#         x1 = self.inputs[1].data
#         return gy * x1, gy * x0


# def mul(x0, x1):
#     x1 = as_array(x1)
#     return Mul()(x0, x1)


# class Neg(Function):
#     def forward(self, x):
#         return -x

#     def backward(self, gy):
#         return -gy


# def neg(x):
#     return Neg()(x)


# class Sub(Function):
#     def forward(self, x0, x1):
#         y = x0 - x1
#         return y

#     def backward(self, gy):
#         return gy, -gy


# def sub(x0, x1):
#     x1 = as_array(x1)
#     return Sub()(x0, x1)


# def rsub(x0, x1):
#     x1 = as_array(x1)
#     return Sub()(x1, x0)


# class Div(Function):
#     def forward(self, x0, x1):
#         y = x0 / x1
#         return y

#     def backward(self, gy):
#         x0 = self.inputs[0].data
#         x1 = self.inputs[1].data
#         gx0 = gy / x1
#         gx1 = gy * (- x0 / (x1 * x1))
#         return gx0, gx1


# def div(x0, x1):
#     x1 = as_array(x1)
#     return Div()(x0, x1)


# def rdiv(x0, x1):
#     x1 = as_array(x1)
#     return Div()(x1, x0)


# class Pow(Function):
#     def __init__(self, c) -> None:
#         if isinstance(c, Variable):
#             self.c = c.data
#         else:
#             self.c = c

#     def forward(self, x):
#         y = x ** self.c
#         return y

#     def backward(self, gy):
#         x = self.inputs[0].data
#         c = self.c

#         gx = c * x ** (c - 1) * gy
#         return gx


# def pow(x0, x1):
#     x1 = as_array(x1)
#     return Pow(x1)(x0)


# def rpow(x0, x1):
#     x1 = np.array(x1)
#     return Pow(x0)(x1)


# if __name__ == "__main__":
#     x = Variable(np.array(2.0))
#     y = x + 2.0
#     print(y)

#     z = 3.0 + x
#     print(z)
