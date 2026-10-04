from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Callable, TypeAlias

from dezero import cuda, Parameter
from dezero import Layer, Model
from dezero.array_type import ArrayType


# =============================================================================
# Optimizer (base class)
# =============================================================================

HookFunction: TypeAlias = Callable[[Sequence[Parameter]], Any]


class Optimizer(ABC):
    def __init__(self) -> None:
        self.target: Layer | Model | None = None
        self.hooks: list[HookFunction] = []

    def setup(self, target: Layer | Model) -> Optimizer:
        self.target = target
        return self

    def update(self) -> None:
        if self.target is None:
            raise ValueError("Optimizer.target is None")

        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.hooks:
            f(params)

        for param in params:
            self.update_one(param)

    @abstractmethod
    def update_one(self, param: Parameter):
        # raise NotImplementedError()
        ...

    def add_hook(self, f: HookFunction) -> None:
        self.hooks.append(f)


# =============================================================================
# Hook functions
# =============================================================================
class WeightDecay:
    def __init__(self, rate: float):
        self.rate = rate

    def __call__(self, params: Sequence[Parameter]):
        for param in params:
            if (param.data is None) or (param.grad is None) or (param.grad.data is None):
                raise ValueError("param.data or param.grad is None")
            param.grad.data += self.rate * param.data


class ClipGrad:
    def __init__(self, max_norm: float):
        self.max_norm = max_norm

    def __call__(self, params: Sequence[Parameter]):
        total_norm = 0.0
        for param in params:
            if (param.data is None) or (param.grad is None) or (param.grad.data is None):
                raise ValueError("param.data or param.grad is None")
            total_norm += (param.grad.data ** 2).sum()
        total_norm = math.sqrt(float(total_norm))

        rate = self.max_norm / (total_norm + 1e-6)
        if rate < 1:
            for param in params:
                if (param.data is None) or (param.grad is None) or (param.grad.data is None):
                    raise ValueError("param.data or param.grad is None")
                param.grad.data *= rate


class FreezeParam:
    def __init__(self, *layers: Parameter | Layer):
        self.freeze_params: list[Parameter] = []
        for layer in layers:
            if isinstance(layer, Parameter):
                self.freeze_params.append(layer)
            else:
                for p in layer.params():
                    self.freeze_params.append(p)

    def __call__(self, params: Sequence[Parameter]):
        for p in self.freeze_params:
            p.grad = None


# =============================================================================
# SGD / MomentumSGD / AdaGrad / AdaDelta / Adam
# =============================================================================

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        super().__init__()
        self.lr = lr

    def update_one(self, param: Parameter) -> None:
        if (param.data is None) or (param.grad is None) or (param.grad.data is None):
            raise ValueError("param.data or param.grad is None")
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs: dict[int, ArrayType] = {}

    def update_one(self, param: Parameter) -> None:
        if (param.data is None) or (param.grad is None) or (param.grad.data is None):
            raise ValueError("param.data or param.grad is None")

        v_key = id(param)
        if v_key not in self.vs:
            xp = cuda.get_array_module(param.data)
            self.vs[v_key] = xp.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v


class AdaGrad(Optimizer):
    def __init__(self, lr: float = 0.001, eps: float = 1e-8) -> None:
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.hs: dict[int, ArrayType] = {}

    def update_one(self, param: Parameter) -> None:
        if (param.data is None) or (param.grad is None) or (param.grad.data is None):
            raise ValueError("param.data or param.grad is None")

        xp = cuda.get_array_module(param.data)

        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = xp.zeros_like(param.data)

        lr = self.lr
        eps = self.eps
        grad = param.grad.data
        h = self.hs[h_key]

        h += grad * grad
        param.data -= lr * grad / (xp.sqrt(h) + eps)


class AdaDelta(Optimizer):
    def __init__(self, rho: float = 0.95, eps: float = 1e-6) -> None:
        super().__init__()
        self.rho = rho
        self.eps = eps
        self.msg: dict[int, ArrayType] = {}
        self.msdx: dict[int, ArrayType] = {}

    def update_one(self, param: Parameter):
        if (param.data is None) or (param.grad is None) or (param.grad.data is None):
            raise ValueError("param.data or param.grad is None")

        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.msg:
            self.msg[key] = xp.zeros_like(param.data)
            self.msdx[key] = xp.zeros_like(param.data)

        msg, msdx = self.msg[key], self.msdx[key]
        rho = self.rho
        eps = self.eps
        grad = param.grad.data

        msg *= rho
        msg += (1 - rho) * grad * grad
        dx = xp.sqrt((msdx + eps) / (msg + eps)) * grad
        msdx *= rho
        msdx += (1 - rho) * dx * dx
        param.data -= dx


class Adam(Optimizer):
    def __init__(
        self,
        alpha: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ) -> None:
        super().__init__()
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms: dict[int, ArrayType] = {}
        self.vs: dict[int, ArrayType] = {}

    def update(self, *args, **kwargs) -> None:
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def lr(self):
        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, param):
        if (param.data is None) or (param.grad is None) or (param.grad.data is None):
            raise ValueError("param.data or param.grad is None")

        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)
            self.vs[key] = xp.zeros_like(param.data)

        m = self.ms[key]
        v = self.vs[key]
        beta1 = self.beta1
        beta2 = self.beta2
        eps = self.eps
        grad = param.grad.data

        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * (grad * grad - v)
        param.data -= self.lr * m / (xp.sqrt(v) + eps)
