from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, TypeAlias

import dezero.functions as F
import dezero.layers as L
from dezero import Layer
from dezero import Variable
from dezero import utils
from dezero.array_type import ArrayType


class Model(Layer):
    def plot(self, *inputs: ArrayType | Variable, to_file: str = 'model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


class Sequential(Model):
    def __init__(self, *layers: Layer):
        super().__init__()
        self.layers: list[Layer] = []
        for i, layer in enumerate(layers):
            setattr(self, 'layer' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x: Variable) -> Variable:
        for layer in self.layers:
            x = layer(x)
        return x


ActivationFunction: TypeAlias = Callable[[ArrayType | Variable], Variable]


class MLP(Model):
    def __init__(
        self,
        fc_output_sizes: Sequence[int],
        activation: ActivationFunction = F.sigmoid
    ) -> None:
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'layer' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x: Variable) -> Variable:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        return self.layers[-1](x)
