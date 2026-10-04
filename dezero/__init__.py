from __future__ import annotations

from dezero.config import Config
from dezero.config import using_config
from dezero.config import no_grad
from dezero.config import test_mode

from dezero.array_type import ArrayType

# step23.pyからstep32.pyまではsimple_coreを利用
from dezero.core_simple import Variable as SimpleVariable
from dezero.core_simple import Function as SimpleFunction
from dezero.core_simple import as_array as simple_as_array
from dezero.core_simple import as_variable as simple_as_variable

from dezero.core import Variable
from dezero.core import Function
from dezero.core import Parameter
from dezero.core import as_array
from dezero.core import as_variable
from dezero.layers import Layer
from dezero.models import Model

# import dezero.functions
# import dezero.utils
from dezero import cuda
from dezero import functions
from dezero import datasets
from dezero import optimizers

# setup_variable()
__version__ = "0.0.1"

__all__ = [
    # array_type
    "ArrayType",
    # config
    "Config",
    "using_config",
    "no_grad",
    "test_mode",
    # core_simple
    "SimpleVariable",
    "SimpleFunction",
    "simple_as_array",
    "simple_as_variable",
    # core
    "Variable",
    "Function",
    "Parameter",
    "as_array",
    "as_variable",
    # layers
    "Layer",
    # models
    "Model",
    # other
    "cuda",
    "functions",
    "datasets",
    "optimizers"
]
