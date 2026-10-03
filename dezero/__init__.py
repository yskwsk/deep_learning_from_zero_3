from __future__ import annotations

from dezero.config import Config
from dezero.config import using_config
from dezero.config import no_grad
from dezero.config import test_mode

# step23.pyからstep32.pyまではsimple_coreを利用
from dezero.core_simple import Variable as SimpleVariable
from dezero.core_simple import Function as SimpleFunction
from dezero.core_simple import as_array as simple_as_array
from dezero.core_simple import as_variable as simple_as_variable

from dezero.core import Variable
from dezero.core import Function
# from dezero.core import using_config
# from dezero.core import no_grad
# from dezero.core import test_mode
# from dezero.core import as_array
# from dezero.core import as_variable
# from dezero.core import setup_variable
# from dezero.core import Config

# import dezero.functions
# import dezero.utils

# setup_variable()
__version__ = "0.0.1"

__all__ = [
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
]
